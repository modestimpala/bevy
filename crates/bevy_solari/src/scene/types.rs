use bevy_asset::Handle;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{component::Component, prelude::ReflectComponent, template::FromTemplate};
use bevy_mesh::Mesh;
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::sync_world::SyncToRenderWorld;
use bevy_transform::components::Transform;
use derive_more::derive::From;

/// A mesh component used for raytracing.
///
/// The mesh used in this component must declare the BLAS its material needs in
/// [`Mesh::raytracing`],
/// use the following set of vertex attributes: `{POSITION, NORMAL, UV_0, TANGENT}`, use [`bevy_mesh::PrimitiveTopology::TriangleList`],
/// and use [`bevy_mesh::Indices::U32`].
///
/// The material used for this entity must be [`MeshMaterial3d<StandardMaterial>`].
#[derive(
    Component, FromTemplate, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From,
)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(MeshMaterial3d<StandardMaterial>, Transform, SyncToRenderWorld)]
pub struct RaytracingMesh3d(pub Handle<Mesh>);

/// What an app knows of how a [`RaytracingMesh3d`] with an emissive material gives off light,
/// where the material and mesh alone would mislead the choice of light samples.
///
/// A mesh whose material has an emissive colour is taken for a light over all of its triangles,
/// as bright as that colour. Where an emissive texture leaves most of the mesh dark, nearly every
/// sample of it lands on a triangle that gives no light. Put the triangles that do first in the
/// mesh's indices, and say here how many they are and how bright.
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Component, Clone, PartialEq)]
pub struct RaytracingEmission {
    /// Only the first this many triangles of the mesh emit light. With none, the mesh is no
    /// light at all.
    pub triangles: u32,
    /// The mean luminance of those triangles: the emissive colour times its texture.
    pub luminance: f32,
}

/// What the render world knows of a raytracing instance beside its mesh and material.
#[derive(Component, Clone, Copy, Default, PartialEq)]
pub struct ExtractedRaytracingTraits {
    pub emission: Option<RaytracingEmission>,
    /// The instance has [`bevy_light::NotShadowCaster`]: rays that ask whether a light can be
    /// seen pass through it, while rays that look for a surface still find it.
    pub shadowless: bool,
}
