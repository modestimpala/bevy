use super::{allocator::SlotAllocator, assets::AssetState, instances::InstanceState};
use crate::scene::blas::BlasManager;
use bevy_color::ColorToComponents;
use bevy_ecs::{
    entity::{Entity, EntityHashSet},
    system::Query,
};
use bevy_math::{ops::cos, Vec3};
use bevy_pbr::ExtractedDirectionalLight;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::render_resource::{AtomicSparseBufferVec, BufferUsages};
use bevy_render::{impl_atomic_pod, render_resource::AtomicPod};
use bytemuck::{Pod, Zeroable};
use core::sync::atomic::{AtomicBool, Ordering};
use core::{f32::consts::TAU, hash::Hash};
use tracing::info_span;

const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;

/// A light sample names its triangle in sixteen bits.
pub const MAX_EMISSIVE_MESH_TRIANGLES: u32 = u16::MAX as u32;

/// The share of light samples spread evenly over the lights. The rest go to each light by how
/// much light it is reckoned to give.
///
/// That reckoning knows nothing of what stands between a light and what it lights, so a light
/// it rates low can still be the one that matters. The even share bounds the harm: no light is
/// picked less than this share of what an even choice would give it.
const EVEN_SHARE: f32 = 0.5;

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuLightSource {
    kind: u32,
    id: u32,
    /// The chance of a light sample picking this light.
    probability: f32,
    /// This slot of the alias table: a sample that lands on it keeps this light with chance
    /// `threshold`, and otherwise takes the light `alias`.
    threshold: f32,
    alias: u32,
}

/// Stable identity for one source in the light array.
///
/// An entity can contribute both kinds at once, so the entity alone is not enough to identify a
/// source.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum LightSourceId {
    EmissiveMesh(Entity),
    Directional(Entity),
}

#[derive(Default)]
pub struct LightIndex {
    indices: HashMap<LightSourceId, u32>,
    ids: Vec<LightSourceId>,
    changed: HashSet<LightSourceId>,
}

impl LightIndex {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn get(&self, id: &LightSourceId) -> Option<u32> {
        self.indices.get(id).copied()
    }

    fn insert(&mut self, id: LightSourceId) -> u32 {
        if let Some(&index) = self.indices.get(&id) {
            return index;
        }

        let index = self.ids.len() as u32;
        self.ids.push(id);
        self.indices.insert(id, index);
        self.changed.insert(id);
        index
    }

    /// Removes `id` and reports both its old index and the old final index.
    ///
    /// Only the ids tracked here are swapped down. When the two indices differ, the caller has to
    /// mirror that swap in `sources`, copying the element at the old final index into the hole.
    fn remove(&mut self, id: LightSourceId) -> Option<(u32, u32)> {
        let index = self.indices.remove(&id)?;
        self.changed.insert(id);

        let last = self.ids.len() as u32 - 1;
        self.ids.swap_remove(index as usize);

        if index != last {
            let moved = self.ids[index as usize];
            self.indices.insert(moved, index);
            self.changed.insert(moved);
        }

        Some((index, last))
    }
}

impl GpuLightSource {
    pub fn new_emissive_mesh_light(instance_id: u32, triangle_count: u32) -> GpuLightSource {
        debug_assert!(triangle_count <= MAX_EMISSIVE_MESH_TRIANGLES);

        Self {
            kind: triangle_count << 1,
            id: instance_id,
            ..Default::default()
        }
    }

    fn new_directional_light(directional_light_id: u32) -> GpuLightSource {
        Self {
            kind: 1,
            id: directional_light_id,
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuDirectionalLight {
    direction_to_light: Vec3,
    cos_theta_max: f32,
    luminance: Vec3,
    inverse_pdf: f32,
}

impl_atomic_pod!(GpuLightSource, GpuLightSourceBlob);
impl_atomic_pod!(GpuDirectionalLight, GpuDirectionalLightBlob);

impl GpuDirectionalLight {
    /// The illuminance of a surface facing the light.
    fn illuminance(&self) -> f32 {
        (self.luminance * self.inverse_pdf).dot(LUMINANCE)
    }

    fn new(directional_light: &ExtractedDirectionalLight) -> Self {
        let cos_theta_max = cos(directional_light.sun_disk_angular_size / 2.0);
        let solid_angle = TAU * (1.0 - cos_theta_max);
        let luminance =
            (directional_light.color.to_vec3() * directional_light.illuminance) / solid_angle;

        Self {
            direction_to_light: directional_light.transform.back().into(),
            cos_theta_max,
            luminance,
            inverse_pdf: solid_angle,
        }
    }
}

/// Light slots and the incremental previous-frame id translation state.
pub struct LightState {
    /// Kept gap-free because shaders derive the light count with `arrayLength`.
    pub sources: AtomicSparseBufferVec<GpuLightSource>,
    pub directional_lights: AtomicSparseBufferVec<GpuDirectionalLight>,
    pub previous_frame_id_translations: AtomicSparseBufferVec<u32>,
    pub index: LightIndex,
    /// Light ids as of the last frame whose translation table the lighting shader actually read.
    previous_index: HashMap<LightSourceId, u32>,
    nonidentity_translations: Vec<u32>,
    directional_slots: SlotAllocator<Entity>,
    /// Set by the lighting node once it has recorded work reading the translation table.
    translations_consumed: AtomicBool,
    /// Scratch for [`Self::weigh`].
    weights: Vec<f32>,
    alias_table: AliasTable,
}

impl LightState {
    pub fn new() -> Self {
        Self {
            sources: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_light_sources".into(),
            ),
            directional_lights: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_directional_lights".into(),
            ),
            previous_frame_id_translations: AtomicSparseBufferVec::new(
                BufferUsages::STORAGE,
                "solari_previous_frame_light_id_translations".into(),
            ),
            index: LightIndex::default(),
            previous_index: HashMap::default(),
            nonidentity_translations: Vec::new(),
            directional_slots: SlotAllocator::new(),
            translations_consumed: AtomicBool::new(false),
            weights: Vec::new(),
            alias_table: AliasTable::default(),
        }
    }

    /// Reckons how much light each source gives the viewers, and from that the chance of a
    /// light sample picking it.
    ///
    /// A directional light is weighed by its illuminance, and an emissive mesh by the
    /// illuminance its surface would give a viewer facing it: luminance times area over distance
    /// squared. Without that a sun among a few hundred small flames gets one sample in a few
    /// hundred.
    pub fn weigh(
        &mut self,
        instances: &InstanceState,
        assets: &AssetState,
        blas_manager: &BlasManager,
        viewers: &[Vec3],
    ) {
        let _span = info_span!("weigh_lights").entered();

        self.weights.clear();
        for (index, id) in self.index.ids.iter().enumerate() {
            let weight = match *id {
                LightSourceId::Directional(_) => {
                    let slot = self.sources.get(index as u32).id;
                    self.directional_lights.get(slot).illuminance()
                }
                LightSourceId::EmissiveMesh(entity) => instances
                    .emitter(entity, assets, blas_manager)
                    .map_or(0.0, |(_, centre, area, luminance)| {
                        let distance_squared = viewers
                            .iter()
                            .map(|viewer| viewer.distance_squared(centre))
                            .reduce(f32::min)
                            .unwrap_or(1.0);
                        // Close to, a surface gives no more than a sky of its own luminance
                        luminance * area / (distance_squared + area).max(f32::MIN_POSITIVE)
                    }),
            };
            self.weights.push(weight);
        }

        self.alias_table.build(&self.weights, EVEN_SHARE);

        for (index, id) in self.index.ids.iter().enumerate() {
            let index = index as u32;
            let (probability, threshold, alias) = self.alias_table.slot(index as usize);
            self.sources.set_if_changed(
                index,
                GpuLightSource {
                    probability,
                    threshold,
                    alias,
                    ..self.sources.get(index)
                },
            );
            if let LightSourceId::EmissiveMesh(entity) = *id
                && let Some((slot, ..)) = instances.emitter(entity, assets, blas_manager)
            {
                instances.set_light_probability(slot, probability);
            }
        }
    }

    pub fn update(&mut self, directional_lights: &Query<(Entity, &ExtractedDirectionalLight)>) {
        // There are few enough directional lights to just walk them every frame
        let _span = info_span!("update_lights").entered();

        let mut live_directional_lights = EntityHashSet::default();
        for (entity, directional_light) in directional_lights {
            live_directional_lights.insert(entity);

            let slot = self.directional_slots.get_or_allocate(entity);
            self.directional_lights
                .grow_and_set(slot, GpuDirectionalLight::new(directional_light));
            self.add_light(
                LightSourceId::Directional(entity),
                GpuLightSource::new_directional_light(slot),
            );
        }

        let stale: Vec<Entity> = self
            .directional_slots
            .keys()
            .copied()
            .filter(|entity| !live_directional_lights.contains(entity))
            .collect();
        for entity in stale {
            self.directional_slots.remove(&entity);
            self.remove_light(LightSourceId::Directional(entity));
        }

        self.write_light_id_translations();

        if self.index.len() > u16::MAX as usize {
            panic!("Too many light sources in the scene, maximum is 65535.");
        }
    }

    pub fn add_light(&mut self, id: LightSourceId, source: GpuLightSource) {
        let index = self.index.insert(id);
        self.sources.grow_and_set(index, source);
    }

    /// Removes a light, moving the last one down into the hole so the array stays gap-free.
    pub fn remove_light(&mut self, id: LightSourceId) {
        let Some((index, last)) = self.index.remove(id) else {
            return;
        };

        if index != last {
            let source = self.sources.get(last);
            self.sources.grow_and_set(index, source);
        }
    }

    /// Rolls the translation table over for a new frame.
    ///
    /// `previous_index` and `changed` only advance once the shader has read the table. The
    /// lighting node bails out while its pipelines compile, and the reservoirs keep the older ids
    /// across such a gap, so the next table has to translate from those instead. `has_consumers`
    /// is false when no view runs Solari lighting, where deferring forever would grow both
    /// without bound.
    pub fn begin_frame(&mut self, has_consumers: bool) {
        for index in core::mem::take(&mut self.nonidentity_translations) {
            self.previous_frame_id_translations
                .grow_and_set(index, index);
        }

        if !has_consumers || self.translations_consumed.swap(false, Ordering::Relaxed) {
            for id in core::mem::take(&mut self.index.changed) {
                match self.index.get(&id) {
                    Some(index) => self.previous_index.insert(id, index),
                    None => self.previous_index.remove(&id),
                };
            }
        }
    }

    /// Records that the lighting shader read this frame's translation table.
    pub fn note_translations_consumed(&self) {
        self.translations_consumed.store(true, Ordering::Relaxed);
    }

    /// Records where each light that moved or disappeared this frame ended up, so that reservoirs
    /// still carrying last frame's light ids can be remapped.
    fn write_light_id_translations(&mut self) {
        for id in &self.index.changed {
            // Lights that first appeared since the last read table have no previous id
            let Some(&previous) = self.previous_index.get(id) else {
                continue;
            };
            let current = self.index.get(id).unwrap_or(LIGHT_NOT_PRESENT_THIS_FRAME);

            if current != previous {
                self.previous_frame_id_translations
                    .grow_and_set(previous, current);
                self.nonidentity_translations.push(previous);
            }
        }

        // Every index the shader might read has to be backed by a real element
        let light_count = self.index.len() as u32;
        let translations = &mut self.previous_frame_id_translations;
        if translations.len() < light_count {
            let start = translations.len();
            translations.grow(light_count);
            for index in start..light_count {
                translations.set(index, index);
            }
        }
    }
}

const LUMINANCE: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// A table that picks one of many things by weight in constant time: pick a slot evenly, keep
/// its own thing with the slot's threshold as the chance, and otherwise take its alias.
#[derive(Default)]
struct AliasTable {
    probabilities: Vec<f32>,
    thresholds: Vec<f32>,
    aliases: Vec<u32>,
    small: Vec<u32>,
    large: Vec<u32>,
}

impl AliasTable {
    /// Builds the table over `weights`, with `even_share` of the choices spread evenly instead.
    fn build(&mut self, weights: &[f32], even_share: f32) {
        let count = weights.len();
        let usable = |weight: &f32| weight.is_finite() && *weight > 0.0;
        let total: f64 = weights
            .iter()
            .filter(|weight| usable(weight))
            .map(|weight| *weight as f64)
            .sum();
        // With nothing to weigh by, every light has the same chance
        let even_share = if total > 0.0 && total.is_finite() {
            even_share as f64
        } else {
            1.0
        };

        self.probabilities.clear();
        self.probabilities.extend(weights.iter().map(|weight| {
            let by_weight = if usable(weight) && even_share < 1.0 {
                *weight as f64 / total
            } else {
                0.0
            };
            (even_share / count as f64 + (1.0 - even_share) * by_weight) as f32
        }));

        // Vose's method: each slot holds an even share of the whole, part its own light's and
        // the rest topped up from a light with more than an even share
        self.thresholds.clear();
        self.thresholds.extend(
            self.probabilities
                .iter()
                .map(|probability| probability * count as f32),
        );
        self.aliases.clear();
        self.aliases.extend(0..count as u32);
        self.small.clear();
        self.large.clear();
        for (index, threshold) in self.thresholds.iter().enumerate() {
            if *threshold < 1.0 {
                self.small.push(index as u32);
            } else {
                self.large.push(index as u32);
            }
        }
        while let (Some(&small), Some(&large)) = (self.small.last(), self.large.last()) {
            self.small.pop();
            self.aliases[small as usize] = large;
            let left = self.thresholds[large as usize] - (1.0 - self.thresholds[small as usize]);
            self.thresholds[large as usize] = left;
            if left < 1.0 {
                self.large.pop();
                self.small.push(large);
            }
        }
        // Whatever is left over is left by rounding, and holds a whole slot
        for index in self.small.drain(..).chain(self.large.drain(..)) {
            self.thresholds[index as usize] = 1.0;
        }
    }

    /// A slot's light's chance of being picked, and the slot's threshold and alias.
    fn slot(&self, index: usize) -> (f32, f32, u32) {
        (
            self.probabilities[index],
            self.thresholds[index],
            self.aliases[index],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{AliasTable, LightIndex, LightSourceId};
    use bevy_ecs::entity::Entity;

    /// The chance the table as built gives each light, summed over the slots.
    fn realised(table: &AliasTable, count: usize) -> Vec<f32> {
        let mut chances = vec![0.0; count];
        for index in 0..count {
            let (_, threshold, alias) = table.slot(index);
            chances[index] += threshold / count as f32;
            chances[alias as usize] += (1.0 - threshold) / count as f32;
        }
        chances
    }

    #[test]
    fn alias_table_picks_each_light_with_the_chance_it_states() {
        let weights = [100_000.0, 3.0, 0.0, 12.0, f32::NAN, 0.5, 40.0];
        let mut table = AliasTable::default();
        table.build(&weights, 0.5);

        let chances = realised(&table, weights.len());
        let mut sum = 0.0;
        for (index, chance) in chances.iter().enumerate() {
            let (stated, threshold, _) = table.slot(index);
            assert!((0.0..=1.0).contains(&threshold));
            assert!(
                (chance - stated).abs() < 1e-5,
                "{index}: {chance} against {stated}"
            );
            // No light is picked less than half as often as an even choice would
            assert!(stated >= 0.5 / weights.len() as f32 - 1e-6);
            sum += stated;
        }
        assert!((sum - 1.0).abs() < 1e-5);
        // The brightest by far takes nearly all of the weighted half
        assert!(table.slot(0).0 > 0.5);
    }

    #[test]
    fn alias_table_is_even_with_nothing_to_weigh_by() {
        let mut table = AliasTable::default();
        table.build(&[0.0; 5], 0.5);
        for index in 0..5 {
            assert_eq!(table.slot(index), (0.2, 1.0, index as u32));
        }
    }

    #[test]
    fn light_index_keeps_sources_on_the_same_entity_independent() {
        let entity = Entity::PLACEHOLDER;
        let emissive = LightSourceId::EmissiveMesh(entity);
        let directional = LightSourceId::Directional(entity);
        let mut lights = LightIndex::default();

        assert_eq!(lights.insert(emissive), 0);
        assert_eq!(lights.insert(directional), 1);
        assert_eq!(lights.insert(emissive), 0);
        assert_eq!(lights.len(), 2);

        assert_eq!(lights.remove(emissive), Some((0, 1)));
        assert_eq!(lights.get(&emissive), None);
        assert_eq!(lights.get(&directional), Some(0));
        assert_eq!(lights.len(), 1);

        assert_eq!(lights.remove(directional), Some((0, 0)));
        assert!(lights.is_empty());
    }
}
