//! A running fox lit only by Solari: its shadow and its light on the ground come from its
//! skinned pose, which Solari skins for itself each frame.
//!
//! Set `SOLARI_SKINNED_SHOTS=path/prefix` to save a screenshot at a few frames and exit.

use bevy::{
    camera::CameraMainTextureUsages,
    mesh::Indices,
    prelude::*,
    render::{
        render_resource::TextureUsages,
        view::screenshot::{save_to_disk, Screenshot},
    },
    solari::prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins},
    world_serialization::WorldInstanceReady,
};

const GLTF_PATH: &str = "models/animated/Fox.glb";
/// Frames at which a screenshot is taken when capturing, then the frame to exit on.
const SHOTS: [u32; 3] = [240, 250, 260];
const EXIT: u32 = 300;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, SolariPlugins))
        .add_systems(Startup, setup)
        .add_systems(Update, capture)
        .run();
}

#[derive(Component)]
struct AnimationToPlay {
    graph: Handle<AnimationGraph>,
    index: AnimationNodeIndex,
}

fn raytraced(mesh: impl Into<Mesh>) -> Mesh {
    mesh.into().with_generated_tangents().unwrap()
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let (graph, index) = AnimationGraph::from_clip(
        asset_server.load(GltfAssetLabel::Animation(2).from_asset(GLTF_PATH)),
    );
    commands
        .spawn((
            AnimationToPlay {
                graph: graphs.add(graph),
                index,
            },
            WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(GLTF_PATH))),
            Transform::from_scale(Vec3::splat(0.02)),
        ))
        .observe(on_fox_ready);

    let floor = meshes.add(raytraced(Plane3d::default().mesh().size(12.0, 12.0)));
    commands.spawn((
        RaytracingMesh3d(floor.clone()),
        Mesh3d(floor),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.8, 0.8),
            perceptual_roughness: 0.9,
            ..default()
        })),
    ));

    // Low, from the side, so the fox's shadow lies long across the floor.
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(4.0, 2.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Transform::from_xyz(-1.0, 4.0, 5.5).looking_at(Vec3::new(-1.5, 0.0, 0.0), Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariLighting::default(),
    ));
}

/// Plays the run, and puts the fox into the raytracing scene.
fn on_fox_ready(
    ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    to_play: Query<&AnimationToPlay>,
    mut players: Query<&mut AnimationPlayer>,
    mesh_query: Query<&Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
) {
    let Ok(to_play) = to_play.get(ready.entity) else {
        return;
    };
    for child in children.iter_descendants(ready.entity) {
        if let Ok(mut player) = players.get_mut(child) {
            player.play(to_play.index).repeat();
            commands
                .entity(child)
                .insert(AnimationGraphHandle(to_play.graph.clone()));
        }
        if let Ok(Mesh3d(handle)) = mesh_query.get(child) {
            commands
                .entity(child)
                .insert(RaytracingMesh3d(handle.clone()));
            let mut mesh = meshes.get_mut(handle).unwrap();
            if !mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0) {
                let count = mesh.count_vertices();
                mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; count]);
            }
            // Solari reads `u32` indices.
            match mesh.indices_mut() {
                Some(indices) if matches!(indices, Indices::U16(_)) => {
                    *indices = Indices::U32(indices.iter().map(|i| i as u32).collect());
                }
                Some(_) => {}
                None => {
                    let count = mesh.count_vertices() as u32;
                    mesh.insert_indices(Indices::U32((0..count).collect()));
                }
            }
        }
    }
}

fn capture(mut commands: Commands, mut frame: Local<u32>, mut exit: MessageWriter<AppExit>) {
    let Ok(prefix) = std::env::var("SOLARI_SKINNED_SHOTS") else {
        return;
    };
    *frame += 1;
    if let Some(i) = SHOTS.iter().position(|&shot| shot == *frame) {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(format!("{prefix}{i}.png")));
    }
    if *frame == EXIT {
        exit.write(AppExit::Success);
    }
}
