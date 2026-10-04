use super::{
    allocator::{IndexAllocator, RetainedBindingArray},
    assets::AssetState,
    lights::{GpuLightSource, LightSourceId, LightState, MAX_EMISSIVE_MESH_TRIANGLES},
    BlasKey, BlasManager, BlasOpacity, RaytracingMesh3d, RaytracingSceneBindings, RaytracingSkins,
};
use crate::scene::types::ExtractedRaytracingTraits;
use bevy_asset::AssetId;
use bevy_ecs::{
    entity::{Entity, EntityHashMap, EntityHashSet},
    query::{Changed, Or, With},
    system::Query,
};
use bevy_math::{Affine3, Affine3Ext, Vec3, Vec4};
use bevy_mesh::Mesh;
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{
    impl_atomic_pod,
    mesh::allocator::MeshAllocator,
    render_resource::{AtomicPod, AtomicSparseBufferVec, Buffer, BufferId, BufferUsages},
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::once;
use bytemuck::{Pod, Zeroable};
use core::{hash::Hash, num::NonZeroU32};
use tracing::{info_span, warn};

pub const MAX_MESH_SLAB_COUNT: NonZeroU32 = NonZeroU32::new(500).unwrap();

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuInstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    /// How many of the mesh's triangles, from its first, a light sample picks among.
    triangle_count: u32,
    /// The chance of a light sample picking this instance, or zero if it is not a light.
    light_probability: f32,
}

/// A world-from-local affine transform, stored transposed as three rows.
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuTransform([Vec4; 3]);

impl GpuTransform {
    /// The three rows as the flat row-major 3x4 a [`TlasInstance`] wants.
    ///
    /// [`TlasInstance`]: bevy_render::render_resource::TlasInstance
    fn rows(self) -> [f32; 12] {
        bytemuck::cast(self)
    }

    /// Where a point of the local space lies in the world.
    fn point(self, local: Vec3) -> Vec3 {
        let local = local.extend(1.0);
        Vec3::new(
            self.0[0].dot(local),
            self.0[1].dot(local),
            self.0[2].dot(local),
        )
    }

    /// About how much larger an area is in the world than in the local space: the mean over the
    /// three axis planes, which is exact for a uniform scale.
    fn area_scale(self) -> f32 {
        let [x, y, z] = [0, 1, 2]
            .map(|axis| Vec3::new(self.0[0][axis], self.0[1][axis], self.0[2][axis]).length());
        (x * y + y * z + z * x) / 3.0
    }
}

/// Every ray finds an instance with this mask.
pub const INSTANCE_MASK_ALL: u32 = 0xFF;
/// Rays that ask whether a light can be seen pass through an instance with this mask. It has to
/// agree with `RAY_CULL_SHADOWLESS` in `bindings.wesl`.
pub const INSTANCE_MASK_SHADOWLESS: u32 = 0x01;

/// The device address of a slot's acceleration structure, and which rays find it. A zero address
/// marks an inactive slot.
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuBlasRef {
    address: u64,
    mask: u32,
    _padding: u32,
}

impl GpuBlasRef {
    const NONE: Self = Self {
        address: 0,
        mask: 0,
        _padding: 0,
    };

    fn new(address: u64, mask: u32) -> Self {
        Self {
            address,
            mask,
            _padding: 0,
        }
    }

    fn is_none(self) -> bool {
        self.address == 0
    }
}

impl_atomic_pod!(GpuInstanceGeometryIds, GpuInstanceGeometryIdsBlob);
impl_atomic_pod!(GpuTransform, GpuTransformBlob);
impl_atomic_pod!(GpuBlasRef, GpuBlasRefBlob);

fn storage_buffer<T: AtomicPod>(label: &str) -> AtomicSparseBufferVec<T> {
    AtomicSparseBufferVec::new(BufferUsages::STORAGE, label.into())
}

/// Everything tracked per raytracing instance.
#[derive(Clone, Copy)]
struct Instance {
    slot: u32,
    mesh: AssetId<Mesh>,
    material: AssetId<StandardMaterial>,
    opacity: BlasOpacity,
    buffers: Option<(BufferId, BufferId)>,
    /// What the app says of how the mesh gives off light, and of whether it casts shadows.
    traits: ExtractedRaytracingTraits,
    /// The share of the mesh's triangles that light samples pick among.
    emitting_share: f32,
}

impl Instance {
    fn blas_key(&self) -> BlasKey {
        BlasKey {
            mesh: self.mesh,
            opacity: self.opacity,
        }
    }
}

/// Stable slots, reverse dependency indices and GPU data owned by raytracing instances.
pub struct InstanceState {
    pub vertex_buffers: RetainedBindingArray<BufferId, Buffer>,
    pub index_buffers: RetainedBindingArray<BufferId, Buffer>,
    pub transforms: AtomicSparseBufferVec<GpuTransform>,
    pub previous_frame_transforms: AtomicSparseBufferVec<GpuTransform>,
    pub geometry_ids: AtomicSparseBufferVec<GpuInstanceGeometryIds>,
    pub material_ids: AtomicSparseBufferVec<u32>,
    pub blas_refs: AtomicSparseBufferVec<GpuBlasRef>,
    pub slots: IndexAllocator,
    records: EntityHashMap<Instance>,
    pub live_count: u32,
    pub pending_refresh: EntityHashSet,
    mesh_instances: HashMap<AssetId<Mesh>, EntityHashSet>,
    pub material_instances: HashMap<AssetId<StandardMaterial>, EntityHashSet>,
}

impl InstanceState {
    pub fn new() -> Self {
        Self {
            vertex_buffers: RetainedBindingArray::new(),
            index_buffers: RetainedBindingArray::new(),
            transforms: storage_buffer("solari_transforms"),
            previous_frame_transforms: storage_buffer("solari_previous_frame_transforms"),
            geometry_ids: storage_buffer("solari_geometry_ids"),
            material_ids: storage_buffer("solari_material_ids"),
            blas_refs: storage_buffer("solari_blas_refs"),
            slots: IndexAllocator::new(),
            records: EntityHashMap::default(),
            live_count: 0,
            pending_refresh: EntityHashSet::default(),
            mesh_instances: HashMap::default(),
            material_instances: HashMap::default(),
        }
    }

    /// Every drawable instance's slot, entity, acceleration structure key, ray mask and
    /// world-from-local transform.
    ///
    /// Only the `wgpu-core` TLAS build path needs this, to fill in the instance descriptors that
    /// the raw path sets up on the GPU. Slots with a null acceleration structure reference are not
    /// currently drawable, and are left out.
    pub fn drawable(&self) -> impl Iterator<Item = (u32, Entity, BlasKey, u8, [f32; 12])> + '_ {
        self.records.iter().filter_map(|(&entity, instance)| {
            let slot = instance.slot;
            let reference = self.blas_refs.get(slot);
            (!reference.is_none()).then(|| {
                (
                    slot,
                    entity,
                    instance.blas_key(),
                    reference.mask as u8,
                    self.transforms.get(slot).rows(),
                )
            })
        })
    }

    /// What a light sample would find of an emissive instance: its slot, where the middle of its
    /// surface lies in the world, its area there and the luminance of its material.
    pub fn emitter(
        &self,
        entity: Entity,
        assets: &AssetState,
        blas_manager: &BlasManager,
    ) -> Option<(u32, Vec3, f32, f32)> {
        let instance = self.records.get(&entity)?;
        let surface = blas_manager.surface(&instance.mesh)?;
        let transform = self.transforms.get(instance.slot);
        Some((
            instance.slot,
            transform.point(surface.centre),
            surface.area * instance.emitting_share * transform.area_scale(),
            match instance.traits.emission {
                Some(emission) => emission.luminance,
                None => assets.emission(self.material_ids.get(instance.slot)),
            },
        ))
    }

    /// Records the chance of a light sample picking the instance in `slot`, for rays that hit it.
    pub fn set_light_probability(&self, slot: u32, light_probability: f32) {
        let geometry_ids = self.geometry_ids.get(slot);
        if geometry_ids.light_probability != light_probability {
            self.geometry_ids.set(
                slot,
                GpuInstanceGeometryIds {
                    light_probability,
                    ..geometry_ids
                },
            );
        }
    }

    /// Points each skinned instance whose acceleration structure was rebuilt this frame at the
    /// new one. An instance not yet resolved picks it up when it is.
    pub fn repoint_skins(&mut self, skins: &RaytracingSkins) {
        for (entity, address) in skins.moved() {
            let Some(slot) = self.records.get(&entity).map(|instance| instance.slot) else {
                continue;
            };
            let reference = self.blas_refs.get(slot);
            if !reference.is_none() {
                self.set_blas_ref(slot, GpuBlasRef::new(address, reference.mask));
            }
        }
    }

    /// Queues every instance using `material_id` to be re-resolved.
    pub fn invalidate_material(&mut self, material_id: AssetId<StandardMaterial>) {
        if let Some(instances) = self.material_instances.get(&material_id) {
            self.pending_refresh.extend(instances.iter().copied());
        }
    }
}

pub type InstanceQueryData<'w> = (
    &'w RaytracingMesh3d,
    &'w MeshMaterial3d<StandardMaterial>,
    &'w GlobalTransform,
    &'w PreviousGlobalTransform,
    Option<&'w ExtractedRaytracingTraits>,
);

pub type ChangedInstanceFilter = (
    With<RaytracingMesh3d>,
    Or<(
        Changed<RaytracingMesh3d>,
        Changed<MeshMaterial3d<StandardMaterial>>,
        Changed<ExtractedRaytracingTraits>,
    )>,
);

/// The scene state an instance resolves its GPU data against.
pub struct InstanceInputs<'a> {
    pub assets: &'a AssetState,
    pub blas_manager: &'a BlasManager,
    pub mesh_allocator: &'a MeshAllocator,
    pub skins: &'a RaytracingSkins,
}

fn unlink<K: Eq + Hash>(map: &mut HashMap<K, EntityHashSet>, key: &K, entity: Entity) {
    let now_empty = map.get_mut(key).is_some_and(|instances| {
        instances.remove(&entity);
        instances.is_empty()
    });
    if now_empty {
        map.remove(key);
    }
}

fn relink<K: Copy + Eq + Hash>(
    map: &mut HashMap<K, EntityHashSet>,
    entity: Entity,
    previous: Option<K>,
    key: K,
) {
    if previous == Some(key) {
        return;
    }
    if let Some(previous) = previous {
        unlink(map, &previous, entity);
    }
    map.entry(key).or_default().insert(entity);
}

impl InstanceState {
    pub fn remove_instances(
        &mut self,
        lights: &mut LightState,
        removed: impl IntoIterator<Item = Entity>,
    ) {
        let _span = info_span!("remove_instances").entered();
        for entity in removed {
            self.remove_instance(lights, entity);
        }
    }

    pub fn refresh_instances(
        &mut self,
        inputs: &InstanceInputs,
        lights: &mut LightState,
        instances: &Query<InstanceQueryData>,
        changed_instances: &Query<Entity, ChangedInstanceFilter>,
    ) {
        let _span = info_span!("refresh_instances").entered();

        let mut refresh = core::mem::take(&mut self.pending_refresh);
        refresh.extend(changed_instances.iter());

        let moved_meshes = inputs.mesh_allocator.meshes_displaced_by_slab_growth();
        for mesh_id in inputs
            .blas_manager
            .changed_meshes()
            .iter()
            .copied()
            .chain(moved_meshes)
        {
            if let Some(mesh_instances) = self.mesh_instances.get(&mesh_id) {
                refresh.extend(mesh_instances.iter().copied());
            }
        }

        for entity in refresh {
            match instances.get(entity) {
                Ok(data) => self.refresh_instance(inputs, lights, entity, data),
                Err(_) => self.remove_instance(lights, entity),
            }
        }
    }

    fn reserve_slot(&mut self, slot: u32) {
        let len = slot + 1;
        self.transforms.grow(len);
        self.previous_frame_transforms.grow(len);
        self.blas_refs.grow(len);
    }

    fn refresh_instance(
        &mut self,
        inputs: &InstanceInputs,
        lights: &mut LightState,
        entity: Entity,
        (mesh, material, transform, previous_frame_transform, traits): InstanceQueryData,
    ) {
        let mesh_id = mesh.id();
        let material_id = material.id();
        let previous = self.records.get(&entity).copied();

        relink(
            &mut self.mesh_instances,
            entity,
            previous.map(|instance| instance.mesh),
            mesh_id,
        );
        relink(
            &mut self.material_instances,
            entity,
            previous.map(|instance| instance.material),
            material_id,
        );

        let slot = match previous {
            Some(previous) => previous.slot,
            None => self.slots.allocate(),
        };
        self.reserve_slot(slot);

        // Seed only once. Later refreshes must not overwrite transforms written by extraction.
        if previous.is_none() {
            self.write_transforms(slot, transform, previous_frame_transform);
        }

        let mut instance = Instance {
            slot,
            mesh: mesh_id,
            material: material_id,
            opacity: BlasOpacity::Opaque,
            buffers: previous.and_then(|instance| instance.buffers),
            traits: traits.copied().unwrap_or_default(),
            emitting_share: 0.0,
        };
        let resolved = self.resolve_instance(inputs, lights, entity, &mut instance);

        self.records.insert(entity, instance);
        if !resolved {
            self.pending_refresh.insert(entity);
        }
    }

    fn resolve_instance(
        &mut self,
        inputs: &InstanceInputs,
        lights: &mut LightState,
        entity: Entity,
        instance: &mut Instance,
    ) -> bool {
        let slot = instance.slot;
        let material_slot = inputs.assets.material_slots.get(&instance.material);

        instance.opacity = if inputs
            .assets
            .non_opaque_materials
            .contains(&instance.material)
        {
            BlasOpacity::NonOpaque
        } else {
            BlasOpacity::Opaque
        };

        // A skinned instance is traced from its own skinned vertices and structure, once it has
        // been skinned; anything else from its mesh's
        let geometry = if inputs.skins.contains(entity) {
            inputs
                .skins
                .geometry(entity)
                .map(|(buffer, address)| (buffer, 0, address))
        } else {
            let blas_key = instance.blas_key();
            let blas_address = inputs.blas_manager.device_address(&blas_key);
            if blas_address.is_none()
                && material_slot.is_some()
                && inputs.blas_manager.is_undeclared(&blas_key)
            {
                once!(warn!(
                    "RaytracingMesh3d entity {entity} uses a material that needs `{flag:?}`, but \
                     `Mesh::raytracing` of mesh {mesh} lacks it. Entities like it will not be \
                     raytraced.",
                    flag = instance.opacity.flag(),
                    mesh = instance.mesh,
                ));
            }
            inputs
                .mesh_allocator
                .mesh_vertex_slice(&instance.mesh)
                .zip(blas_address)
                .map(|(slice, address)| (slice.buffer, slice.range.start, address))
        };
        let (
            Some((vertex_buffer, vertex_buffer_offset, blas_address)),
            Some(index_slice),
            Some(material_slot),
        ) = (
            geometry,
            inputs.mesh_allocator.mesh_index_slice(&instance.mesh),
            material_slot,
        )
        else {
            self.deactivate_instance(lights, entity, instance);
            return false;
        };

        let vertex_buffer_key = vertex_buffer.id();
        let index_buffer_key = index_slice.buffer.id();
        let capacity = MAX_MESH_SLAB_COUNT.get();
        if !self.vertex_buffers.has_room(&vertex_buffer_key, capacity)
            || !self.index_buffers.has_room(&index_buffer_key, capacity)
        {
            once!(warn!(
                "Solari scene needs more than {} mesh slabs. Instances past that limit will \
                 not be rendered.",
                MAX_MESH_SLAB_COUNT.get()
            ));
            self.deactivate_instance(lights, entity, instance);
            return false;
        }

        let previous_buffers = instance.buffers.take();
        let vertex_buffer_id = self
            .vertex_buffers
            .acquire(vertex_buffer_key, capacity, || vertex_buffer.clone())
            .expect("vertex slab binding array had room but handed out no slot");
        let index_buffer_id = self
            .index_buffers
            .acquire(index_buffer_key, capacity, || index_slice.buffer.clone())
            .expect("index slab binding array had room but handed out no slot");
        instance.buffers = Some((vertex_buffer_key, index_buffer_key));
        self.release_buffers(previous_buffers);

        let mesh_triangle_count = (index_slice.range.len() / 3) as u32;
        let is_emissive = inputs
            .assets
            .emissive_materials
            .contains(&instance.material);
        let emitting_triangle_count = match instance.traits.emission {
            Some(emission) => emission.triangles.min(mesh_triangle_count),
            None => mesh_triangle_count,
        };
        if is_emissive && emitting_triangle_count > MAX_EMISSIVE_MESH_TRIANGLES {
            once!(warn!(
                "RaytracingMesh3d entity {entity} has an emissive material on a mesh of \
                 {emitting_triangle_count} emitting triangles, and at most \
                 {MAX_EMISSIVE_MESH_TRIANGLES} can be sampled as a light. The rest of meshes like \
                 it only light what rays find by chance."
            ));
        }
        let triangle_count = emitting_triangle_count.min(MAX_EMISSIVE_MESH_TRIANGLES);
        instance.emitting_share = triangle_count as f32 / mesh_triangle_count.max(1) as f32;
        self.geometry_ids.grow_and_set(
            slot,
            GpuInstanceGeometryIds {
                vertex_buffer_id,
                vertex_buffer_offset,
                index_buffer_id,
                index_buffer_offset: index_slice.range.start,
                triangle_count,
                // Written once the lights have been weighed against each other
                light_probability: 0.0,
            },
        );
        self.material_ids.grow_and_set(slot, material_slot);
        let mask = if instance.traits.shadowless {
            INSTANCE_MASK_SHADOWLESS
        } else {
            INSTANCE_MASK_ALL
        };
        self.set_blas_ref(slot, GpuBlasRef::new(blas_address, mask));

        if is_emissive && triangle_count > 0 {
            lights.add_light(
                LightSourceId::EmissiveMesh(entity),
                GpuLightSource::new_emissive_mesh_light(slot, triangle_count),
            );
        } else {
            lights.remove_light(LightSourceId::EmissiveMesh(entity));
        }
        true
    }

    fn write_transforms(
        &self,
        slot: u32,
        transform: &GlobalTransform,
        previous_frame_transform: &PreviousGlobalTransform,
    ) {
        self.transforms.set_if_changed(
            slot,
            GpuTransform(Affine3::from(transform.affine()).to_transpose()),
        );
        self.previous_frame_transforms.set_if_changed(
            slot,
            GpuTransform(Affine3::from(previous_frame_transform.0).to_transpose()),
        );
    }

    /// Points a slot at an acceleration structure, or at nothing, keeping `live_count` in step.
    fn set_blas_ref(&mut self, slot: u32, reference: GpuBlasRef) {
        self.blas_refs.grow(slot + 1);
        let previous = self.blas_refs.get(slot);
        if previous == reference {
            return;
        }
        self.blas_refs.set(slot, reference);

        if previous.is_none() && !reference.is_none() {
            self.live_count += 1;
        } else if !previous.is_none() && reference.is_none() {
            self.live_count -= 1;
        }
    }

    fn deactivate_instance(
        &mut self,
        lights: &mut LightState,
        entity: Entity,
        instance: &mut Instance,
    ) {
        self.set_blas_ref(instance.slot, GpuBlasRef::NONE);
        lights.remove_light(LightSourceId::EmissiveMesh(entity));
        self.release_buffers(instance.buffers.take());
    }

    fn release_buffers(&mut self, buffers: Option<(BufferId, BufferId)>) {
        if let Some((vertex_key, index_key)) = buffers {
            self.vertex_buffers.release(&vertex_key);
            self.index_buffers.release(&index_key);
        }
    }

    fn remove_instance(&mut self, lights: &mut LightState, entity: Entity) {
        let Some(mut instance) = self.records.remove(&entity) else {
            return;
        };

        self.deactivate_instance(lights, entity, &mut instance);
        self.slots.release(instance.slot);
        self.pending_refresh.remove(&entity);
        unlink(&mut self.mesh_instances, &instance.mesh, entity);
        unlink(&mut self.material_instances, &instance.material, entity);
    }
}

impl RaytracingSceneBindings {
    /// Parallel hot path: one entity lookup, then two allocation-free sparse writes.
    pub fn move_instance(
        &self,
        entity: Entity,
        transform: &GlobalTransform,
        previous_frame_transform: &PreviousGlobalTransform,
    ) {
        if let Some(instance) = self.instances.records.get(&entity) {
            self.instances
                .write_transforms(instance.slot, transform, previous_frame_transform);
        }
    }
}
