// Skins a mesh for raytracing: reads each vertex from the mesh's own slab, blends it by its
// joints as `bevy_pbr::skinning::skin_model` does, brings it back into the instance's local
// space, and writes it in Solari's packed vertex layout.

struct Params {
    // The instance's transform and its inverse, so skinned vertices (which come out in world
    // space) can go back under the transform the TLAS instance applies.
    local_from_world: mat4x4<f32>,
    world_from_local: mat4x4<f32>,
    vertex_count: u32,
    // Where this mesh's first vertex starts in the slab, and each vertex's stride, in words.
    first_word: u32,
    stride_words: u32,
    // The skin's first joint matrix.
    joint_offset: u32,
    // Each attribute's word within a vertex; `ABSENT` when the mesh has none.
    position_word: u32,
    normal_word: u32,
    uv_word: u32,
    tangent_word: u32,
    joint_index_word: u32,
    joint_weight_word: u32,
    _pad0: u32,
    _pad1: u32,
}

const ABSENT: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read_write> skinned: array<vec4<f32>>;

fn read_vec2(word: u32) -> vec2<f32> {
    return vec2(bitcast<f32>(source[word]), bitcast<f32>(source[word + 1u]));
}

fn read_vec3(word: u32) -> vec3<f32> {
    return vec3(read_vec2(word), bitcast<f32>(source[word + 2u]));
}

fn read_vec4(word: u32) -> vec4<f32> {
    return vec4(read_vec3(word), bitcast<f32>(source[word + 3u]));
}

fn inverse_transpose_3x3(m: mat3x3<f32>) -> mat3x3<f32> {
    let x = cross(m[1], m[2]);
    let y = cross(m[2], m[0]);
    let z = cross(m[0], m[1]);
    let det = dot(m[2], z);
    return mat3x3<f32>(x / det, y / det, z / det);
}

fn upper(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}

fn any_orthonormal(n: vec3<f32>) -> vec3<f32> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    return vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
}

@compute @workgroup_size(64, 1, 1)
fn skin(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vertex = global_id.x;
    if vertex >= params.vertex_count {
        return;
    }
    let base = params.first_word + vertex * params.stride_words;

    // Joint indices are `Uint16x4`: two to a word, low half first.
    let packed_lo = source[base + params.joint_index_word];
    let packed_hi = source[base + params.joint_index_word + 1u];
    let joints = vec4(
        packed_lo & 0xFFFFu,
        packed_lo >> 16u,
        packed_hi & 0xFFFFu,
        packed_hi >> 16u,
    ) + vec4(params.joint_offset);
    let weights = read_vec4(base + params.joint_weight_word);
    let world_from_skin = weights.x * joint_matrices[joints.x]
        + weights.y * joint_matrices[joints.y]
        + weights.z * joint_matrices[joints.z]
        + weights.w * joint_matrices[joints.w];

    // Into world space as raster has it, then back under the instance's transform.
    let world_position = world_from_skin * vec4(read_vec3(base + params.position_word), 1.0);
    let position = (params.local_from_world * world_position).xyz;

    let world_normal = inverse_transpose_3x3(upper(world_from_skin))
        * read_vec3(base + params.normal_word);
    // A normal goes from world to local by the transpose of the local-to-world matrix.
    let normal = normalize(transpose(upper(params.world_from_local)) * world_normal);

    var uv = vec2(0.0);
    if params.uv_word != ABSENT {
        uv = read_vec2(base + params.uv_word);
    }

    var tangent: vec4<f32>;
    if params.tangent_word != ABSENT {
        let rest = read_vec4(base + params.tangent_word);
        let world_tangent = upper(world_from_skin) * rest.xyz;
        tangent = vec4(normalize(upper(params.local_from_world) * world_tangent), rest.w);
    } else {
        tangent = vec4(any_orthonormal(normal), 1.0);
    }

    let out = vertex * 3u;
    skinned[out] = vec4(position, normal.x);
    skinned[out + 1u] = vec4(normal.yz, uv);
    skinned[out + 2u] = tangent;
}
