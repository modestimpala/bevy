import bevy_pbr::deferred::types::unpack_24bit_normal;
import bevy_pbr::render::rgb9e5::rgb9e5_to_vec3_;
import bevy_render::utils::octahedral_decode;
import bevy_render::view::{View, depth_ndc_to_view_z};
import package::scene::bindings::ResolvedMaterial;

struct ResolvedGPixel {
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
}

fn gpixel_resolve(gpixel: vec4<u32>, depth: f32, pixel_id: vec2<u32>, view_size: vec2<f32>, world_from_clip: mat4x4<f32>) -> ResolvedGPixel {
    let world_position = reconstruct_world_position(pixel_id, depth, view_size, world_from_clip);
    let world_normal = octahedral_decode(unpack_24bit_normal(gpixel.a));

    let base_rough = unpack4x8unorm(gpixel.r);
    let base_color = pow(base_rough.rgb, vec3(2.2));
    let perceptual_roughness = base_rough.a;
    let roughness = perceptual_roughness * perceptual_roughness;
    let props = unpack4x8unorm(gpixel.b);
    let reflectance = props.r;
    let metallic = props.g;
    let emissive = rgb9e5_to_vec3_(gpixel.g);
    let material = ResolvedMaterial(base_color, emissive, reflectance, perceptual_roughness, roughness, metallic);

    return ResolvedGPixel(world_position, world_normal, material);
}

fn reconstruct_world_position(pixel_id: vec2<u32>, depth: f32, view_size: vec2<f32>, world_from_clip: mat4x4<f32>) -> vec3<f32> {
    let uv = (vec2<f32>(pixel_id) + 0.5) / view_size;
    let xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    let world_pos = world_from_clip * vec4(xy_ndc, depth, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Reject if tangent plane differs more than 0.3% or angle between normals more than 90 degrees
fn pixel_dissimilar(depth: f32, world_position: vec3<f32>, other_world_position: vec3<f32>, normal: vec3<f32>, other_normal: vec3<f32>, view: View) -> bool {
    // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf#page=45
    let tangent_plane_distance = abs(dot(normal, other_world_position - world_position));
    let view_z = -depth_ndc_to_view_z(depth, view.clip_from_view, view.view_from_clip);

    return tangent_plane_distance / view_z > 0.003 || dot(normal, other_normal) < 0.0;
}

/// As [`pixel_dissimilar`], of a surface that may have moved of itself since the other
/// was seen. `moved` is how far it has, across the view, in world units: a surface that
/// moves across the view moves along its normal about as far, and where it was seen last
/// is that far from where it is now. Without the allowance whatever moves is found
/// dissimilar to itself every frame, and is lit from nothing but this frame's samples.
fn pixel_dissimilar_moved(depth: f32, world_position: vec3<f32>, other_world_position: vec3<f32>, normal: vec3<f32>, other_normal: vec3<f32>, view: View, moved: f32) -> bool {
    let tangent_plane_distance = abs(dot(normal, other_world_position - world_position));
    let view_z = -depth_ndc_to_view_z(depth, view.clip_from_view, view.view_from_clip);

    return tangent_plane_distance > 0.003 * view_z + 2.0 * moved || dot(normal, other_normal) < 0.0;
}

/// How far a surface has moved of itself across the view since the last frame, in world
/// units: what its motion vector says, less what the camera's own motion accounts for.
/// The first pixel of it is not counted, which is within what the vectors are good to.
fn moved_of_itself(depth: f32, world_position: vec3<f32>, motion_vector: vec2<f32>, view: View, previous_unjittered_clip_from_world: mat4x4<f32>) -> f32 {
    let now = view.unjittered_clip_from_world * vec4(world_position, 1.0);
    let before = previous_unjittered_clip_from_world * vec4(world_position, 1.0);
    if now.w <= 0.0 || before.w <= 0.0 {
        return 0.0;
    }
    let camera_motion = (now.xy / now.w - before.xy / before.w) * vec2(0.5, -0.5);
    let pixels = length((motion_vector - camera_motion) * view.main_pass_viewport.zw);
    let view_z = -depth_ndc_to_view_z(depth, view.clip_from_view, view.view_from_clip);
    let world_per_pixel = 2.0 * view_z / (view.clip_from_view[1][1] * view.main_pass_viewport.w);
    return max(pixels - 1.0, 0.0) * world_per_pixel;
}

fn permute_pixel(pixel_id: vec2<u32>, frame_index: u32, view_size: vec2<f32>) -> vec2<u32> {
    let r = frame_index;
    let offset = vec2<i32>(i32(r & 3u), i32((r >> 2u) & 3u));
    var shifted_pixel_id = vec2<i32>(pixel_id) + offset;
    shifted_pixel_id ^= vec2<i32>(3);
    shifted_pixel_id -= offset;
    if any(shifted_pixel_id < vec2<i32>(0)) || any(shifted_pixel_id >= vec2<i32>(view_size)) {
        return pixel_id;
    }
    return vec2<u32>(shifted_pixel_id);
}
