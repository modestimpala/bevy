//! Skinned meshes in the raytracing scene.
//!
//! Bevy skins in its vertex shaders, so a skinned mesh's posed vertices never exist in a buffer
//! that an acceleration structure could be built from. Solari skins them for itself: each
//! skinned instance is given its own vertex buffer, which a compute pass fills from the mesh's
//! slab and the skin's joint matrices (those `bevy_pbr` has already uploaded for raster), and
//! its own acceleration structures, rebuilt from that buffer whenever the skin moves.
//!
//! The vertices are brought back into the instance's local space, so the TLAS instance keeps
//! the entity's transform like any other. Hits read the instance's own buffer and the mesh's
//! index slice as usual.
//!
//! Each instance keeps [`BLAS_RING`] acceleration structures and builds into the oldest. The
//! TLAS is double buffered, and a structure may only be rebuilt once two TLAS builds have passed
//! since one last referred to it, as for deletion (see `blas.rs`).
//!
//! While a skin is off screen, Bevy drops its joints; the instance keeps its last pose until it
//! is seen again. Morph targets are not applied.

use super::{blas::BlasManager, RaytracingMesh3d, RaytracingSceneBindings};
use bevy_asset::{load_embedded_asset, AssetId};
use bevy_ecs::{
    component::Component,
    entity::{Entity, EntityHashMap, EntityHashSet},
    query::With,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::{Affine3A, Mat4};
use bevy_mesh::{skinning::SkinnedMesh, Mesh, MeshVertexAttribute, VertexFormat};
use bevy_pbr::SkinUniforms;
use bevy_render::{
    diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
        },
        *,
    },
    renderer::{PendingCommandBuffers, RenderDevice, RenderQueue},
    sync_world::{MainEntity, RenderEntity},
    Extract,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::{default, once};
use bytemuck::{Pod, Zeroable};
use core::num::NonZeroU64;
use tracing::warn;

/// Acceleration structures kept per skinned instance, built into in turn.
const BLAS_RING: usize = 3;
/// Bytes of one vertex in Solari's packed layout: position and normal, normal and UV, tangent.
const PACKED_VERTEX_BYTES: u64 = 48;
/// Marks an attribute the mesh does not have.
const ABSENT: u32 = u32::MAX;

/// A skinned raytracing instance's skin, as extracted this frame.
#[derive(Component, Clone, Copy)]
pub struct ExtractedRaytracingSkin {
    joints: u32,
    world_from_local: Affine3A,
}

/// Extracts every skinned raytracing instance's joint count and transform.
pub fn extract_raytracing_skins(
    skins: Extract<Query<(RenderEntity, &SkinnedMesh, &GlobalTransform), With<RaytracingMesh3d>>>,
    mut commands: Commands,
) {
    for (render_entity, skin, transform) in &skins {
        commands
            .entity(render_entity)
            .insert(ExtractedRaytracingSkin {
                joints: skin.joints.len() as u32,
                world_from_local: transform.affine(),
            });
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuSkinParams {
    local_from_world: Mat4,
    world_from_local: Mat4,
    vertex_count: u32,
    first_word: u32,
    stride_words: u32,
    joint_offset: u32,
    position_word: u32,
    normal_word: u32,
    uv_word: u32,
    tangent_word: u32,
    joint_index_word: u32,
    joint_weight_word: u32,
    _pad: [u32; 2],
}

/// Where each attribute a skinned vertex is read from lies within a vertex, in words.
#[derive(Clone, Copy, PartialEq)]
struct SourceLayout {
    stride_words: u32,
    position: u32,
    normal: u32,
    uv: u32,
    tangent: u32,
    joint_index: u32,
    joint_weight: u32,
}

impl SourceLayout {
    /// The layout of a skinned mesh Solari can skin, or why it cannot.
    fn of(mesh: &RenderMesh) -> Result<Self, &'static str> {
        let layout = &mesh.layout.0;
        let buffer = layout.layout();
        let find = |attribute: &MeshVertexAttribute| -> Result<Option<u32>, &'static str> {
            let Some(index) = layout
                .attribute_ids()
                .iter()
                .position(|id| *id == attribute.id)
            else {
                return Ok(None);
            };
            let found = &buffer.attributes[index];
            if found.format != attribute.format {
                return Err("an attribute is compressed or not in its usual format");
            }
            if !found.offset.is_multiple_of(4) {
                return Err("an attribute is not word aligned");
            }
            Ok(Some((found.offset / 4) as u32))
        };
        let needed = |attribute| find(attribute)?.ok_or("a needed attribute is missing");
        if !buffer.array_stride.is_multiple_of(4) {
            return Err("the vertex stride is not whole words");
        }
        if !matches!(
            mesh.buffer_info,
            RenderMeshBufferInfo::Indexed {
                index_format: IndexFormat::Uint32,
                ..
            }
        ) {
            return Err("the mesh is not indexed with u32 indices");
        }
        if Mesh::ATTRIBUTE_JOINT_INDEX.format != VertexFormat::Uint16x4 {
            return Err("joint indices are not Uint16x4");
        }
        Ok(Self {
            stride_words: (buffer.array_stride / 4) as u32,
            position: needed(&Mesh::ATTRIBUTE_POSITION)?,
            normal: needed(&Mesh::ATTRIBUTE_NORMAL)?,
            uv: find(&Mesh::ATTRIBUTE_UV_0)?.unwrap_or(ABSENT),
            tangent: find(&Mesh::ATTRIBUTE_TANGENT)?.unwrap_or(ABSENT),
            joint_index: needed(&Mesh::ATTRIBUTE_JOINT_INDEX)?,
            joint_weight: needed(&Mesh::ATTRIBUTE_JOINT_WEIGHT)?,
        })
    }
}

/// One skinned instance's buffers and acceleration structures.
struct SkinnedInstance {
    mesh: AssetId<Mesh>,
    source: SourceLayout,
    vertex_count: u32,
    index_count: u32,
    output: Buffer,
    params: Buffer,
    blas: [Blas; BLAS_RING],
    blas_size: BlasTriangleGeometrySizeDescriptor,
    /// The structure last built, if any has been.
    current: Option<usize>,
    /// The joints and transform it was last skinned with.
    joints: Vec<Mat4>,
    world_from_local: Affine3A,
}

/// Every skinned instance in the raytracing scene, and the pass that skins them.
#[derive(Resource)]
pub struct RaytracingSkins {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    instances: EntityHashMap<SkinnedInstance>,
    /// Instances whose acceleration structure changed this frame.
    moved: Vec<Entity>,
}

impl FromWorld for RaytracingSkins {
    fn from_world(world: &mut World) -> Self {
        let layout = BindGroupLayoutDescriptor::new(
            "solari_skinning_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, NonZeroU64::new(size_of::<GpuSkinParams>() as u64)),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let shader = load_embedded_asset!(world, "skinning.wesl");
        let pipeline =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("solari_skinning_pipeline".into()),
                    layout: vec![layout.clone()],
                    shader,
                    entry_point: Some("skin".into()),
                    ..default()
                });
        Self {
            layout,
            pipeline,
            instances: EntityHashMap::default(),
            moved: Vec::new(),
        }
    }
}

impl RaytracingSkins {
    /// A skinned instance's own vertex buffer and its latest acceleration structure's address,
    /// once it has been skinned.
    pub(crate) fn geometry(&self, entity: Entity) -> Option<(&Buffer, u64)> {
        let instance = self.instances.get(&entity)?;
        let blas = &instance.blas[instance.current?];
        Some((&instance.output, blas.handle()?))
    }

    /// Whether an instance is skinned here, whether or not it has been skinned yet.
    pub(crate) fn contains(&self, entity: Entity) -> bool {
        self.instances.contains_key(&entity)
    }

    /// A skinned instance's latest acceleration structure.
    pub(crate) fn blas(&self, entity: Entity) -> Option<&Blas> {
        let instance = self.instances.get(&entity)?;
        Some(&instance.blas[instance.current?])
    }

    /// Instances whose acceleration structure changed this frame, and its address.
    pub(crate) fn moved(&self) -> impl Iterator<Item = (Entity, u64)> + '_ {
        self.moved
            .iter()
            .filter_map(|&entity| Some((entity, self.geometry(entity)?.1)))
    }
}

fn create_instance(
    render_device: &RenderDevice,
    mesh: AssetId<Mesh>,
    source: SourceLayout,
    vertex_count: u32,
    index_count: u32,
) -> SkinnedInstance {
    let output = render_device.create_buffer(&BufferDescriptor {
        label: Some("solari_skinned_vertices"),
        size: vertex_count as u64 * PACKED_VERTEX_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::BLAS_INPUT,
        mapped_at_creation: false,
    });
    let params = render_device.create_buffer(&BufferDescriptor {
        label: Some("solari_skinning_params"),
        size: size_of::<GpuSkinParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: Mesh::ATTRIBUTE_POSITION.format,
        vertex_count,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(index_count),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = core::array::from_fn(|_| {
        render_device.wgpu_device().create_blas(
            &CreateBlasDescriptor {
                label: Some("solari_skinned_blas"),
                // Rebuilt whenever the skin moves, which is most frames: a quick build is
                // worth more than a quicker trace.
                flags: AccelerationStructureFlags::PREFER_FAST_BUILD,
                update_mode: AccelerationStructureUpdateMode::Build,
            },
            BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![blas_size.clone()],
            },
        )
    });
    SkinnedInstance {
        mesh,
        source,
        vertex_count,
        index_count,
        output,
        params,
        blas,
        blas_size,
        current: None,
        joints: Vec::new(),
        world_from_local: Affine3A::IDENTITY,
    }
}

/// Skins every skinned raytracing instance whose skin moved, and rebuilds its acceleration
/// structure.
///
/// Runs after `bevy_pbr` has uploaded this frame's joints and before the scene resolves its
/// instances, which read the buffers and structures made here.
pub fn prepare_raytracing_skins(
    instances: Query<(
        Entity,
        &MainEntity,
        &RaytracingMesh3d,
        Option<&ExtractedRaytracingSkin>,
    )>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    skin_uniforms: Res<SkinUniforms>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut pending_command_buffers: ResMut<PendingCommandBuffers>,
    mut skins: ResMut<RaytracingSkins>,
    mut blas_manager: ResMut<BlasManager>,
    mut bindings: ResMut<RaytracingSceneBindings>,
    mut diagnostics: Option<ResMut<DiagnosticsRecorder>>,
) {
    let skins = &mut *skins;
    skins.moved.clear();
    let pipeline = pipeline_cache.get_compute_pipeline(skins.pipeline);
    let layout = pipeline_cache.get_bind_group_layout(&skins.layout);

    let mut seen = EntityHashSet::default();
    // (entity, bind group, workgroups, index slab and its first index)
    let mut work = Vec::new();

    for (entity, main_entity, mesh, skin) in &instances {
        let Some(skin) = skin else {
            continue;
        };
        let mesh_id = mesh.id();
        let Some(render_mesh) = render_meshes.get(mesh_id) else {
            continue;
        };
        if !render_mesh.layout.0.contains(Mesh::ATTRIBUTE_JOINT_INDEX) {
            continue;
        }
        let source = match SourceLayout::of(render_mesh) {
            Ok(source) => source,
            Err(why) => {
                once!(warn!(
                    "A skinned mesh cannot be raytraced and is left out of the scene: {why}."
                ));
                continue;
            }
        };
        let (Some(vertex_slice), Some(index_slice)) = (
            mesh_allocator.mesh_vertex_slice(&mesh_id),
            mesh_allocator.mesh_index_slice(&mesh_id),
        ) else {
            continue;
        };
        seen.insert(entity);

        let vertex_count = vertex_slice.range.len() as u32;
        let index_count = index_slice.range.len() as u32;
        let stale = skins.instances.get(&entity).is_none_or(|instance| {
            instance.mesh != mesh_id
                || instance.source != source
                || instance.vertex_count != vertex_count
                || instance.index_count != index_count
        });
        if stale {
            let fresh = create_instance(&render_device, mesh_id, source, vertex_count, index_count);
            if let Some(old) = skins.instances.insert(entity, fresh) {
                for blas in old.blas {
                    blas_manager.retire_blas(blas);
                }
            }
            bindings.refresh_instance_later(entity);
        }
        let instance = skins.instances.get_mut(&entity).unwrap();

        // Off screen, Bevy has no joints for it: it keeps its last pose.
        let Some(joint_offset) = skin_uniforms.skin_index(*main_entity) else {
            continue;
        };
        let (Some(pipeline), Some(joints)) = (
            pipeline,
            skin_uniforms
                .current_staging_buffer
                .get(joint_offset as usize..(joint_offset + skin.joints) as usize),
        ) else {
            continue;
        };
        if instance.current.is_some()
            && instance.joints == joints
            && instance.world_from_local == skin.world_from_local
        {
            continue;
        }
        instance.joints.clear();
        instance.joints.extend_from_slice(joints);
        instance.world_from_local = skin.world_from_local;

        let world_from_local = Mat4::from(skin.world_from_local);
        let params = GpuSkinParams {
            local_from_world: world_from_local.inverse(),
            world_from_local,
            vertex_count,
            first_word: vertex_slice.range.start * source.stride_words,
            stride_words: source.stride_words,
            joint_offset,
            position_word: source.position,
            normal_word: source.normal,
            uv_word: source.uv,
            tangent_word: source.tangent,
            joint_index_word: source.joint_index,
            joint_weight_word: source.joint_weight,
            _pad: [0; 2],
        };
        render_queue.write_buffer(&instance.params, 0, bytemuck::bytes_of(&params));
        let bind_group = render_device.create_bind_group(
            "solari_skinning_bind_group",
            &layout,
            &BindGroupEntries::sequential((
                instance.params.as_entire_binding(),
                vertex_slice.buffer.as_entire_binding(),
                skin_uniforms.current_buffer.as_entire_binding(),
                instance.output.as_entire_binding(),
            )),
        );
        let next = instance
            .current
            .map_or(0, |current| (current + 1) % BLAS_RING);
        work.push((
            entity,
            pipeline,
            bind_group,
            vertex_count.div_ceil(64),
            index_slice.buffer.clone(),
            index_slice.range.start,
            next,
        ));
    }

    // Instances no longer skinned here go, and are resolved again against whatever they are now
    let gone: Vec<Entity> = skins
        .instances
        .keys()
        .copied()
        .filter(|entity| !seen.contains(entity))
        .collect();
    for entity in gone {
        if let Some(old) = skins.instances.remove(&entity) {
            for blas in old.blas {
                blas_manager.retire_blas(blas);
            }
        }
        bindings.refresh_instance_later(entity);
    }

    if work.is_empty() {
        return;
    }

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("solari_skinning_command_encoder"),
    });
    if let Some(diagnostics) = diagnostics.as_mut() {
        let time_span = diagnostics.time_span(&mut command_encoder, "solari_skinning");
        skin(&mut command_encoder, &work);
        time_span.end(&mut command_encoder);
    } else {
        skin(&mut command_encoder, &work);
    }
    let geometries: Vec<_> = work
        .iter()
        .map(|(entity, _, _, _, index_buffer, first_index, next)| {
            let instance = &skins.instances[entity];
            (
                &instance.blas[*next],
                BlasTriangleGeometry {
                    size: &instance.blas_size,
                    vertex_buffer: &instance.output,
                    first_vertex: 0,
                    vertex_stride: PACKED_VERTEX_BYTES,
                    index_buffer: Some(index_buffer),
                    first_index: Some(*first_index),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                },
            )
        })
        .collect();
    let entries: Vec<_> = geometries
        .into_iter()
        .map(|(blas, geometry)| BlasBuildEntry {
            blas,
            geometry: BlasGeometries::TriangleGeometries(vec![geometry]),
        })
        .collect();
    if let Some(diagnostics) = diagnostics.as_mut() {
        let time_span = diagnostics.time_span(&mut command_encoder, "skinned_blas_build");
        command_encoder.build_acceleration_structures(&entries, &[]);
        time_span.end(&mut command_encoder);
    } else {
        command_encoder.build_acceleration_structures(&entries, &[]);
    }
    drop(entries);
    // Submitted with the frame's passes, and ahead of them: a submission of its own every
    // frame that anything skinned moves costs the renderer more than the skinning does.
    pending_command_buffers.push_encoder(command_encoder, "solari_skinning");

    for (entity, .., next) in work {
        let instance = skins.instances.get_mut(&entity).unwrap();
        if instance.current.is_none() {
            // Skinned for the first time: the instance can resolve now
            bindings.refresh_instance_later(entity);
        }
        instance.current = Some(next);
        skins.moved.push(entity);
    }
}

/// Records the skinning dispatches of this frame's moved instances.
fn skin(
    command_encoder: &mut CommandEncoder,
    work: &[(Entity, &ComputePipeline, BindGroup, u32, Buffer, u32, usize)],
) {
    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("solari_skinning"),
        timestamp_writes: None,
    });
    for (_, pipeline, bind_group, workgroups, ..) in work {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(*workgroups, 1, 1);
    }
}
