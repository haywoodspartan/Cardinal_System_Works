#pragma once

// =============================================================================
// Cardinal — Skeletal skinning core.
//
// The data model + CPU reference kernel for skinned (animated character)
// meshes. Everything above this (glTF/FBX import, the asset codec, the GPU
// SkinPass in cardinal::render, the forward renderer's per-frame skin step,
// the anim system's pose drivers) speaks these types:
//
//   Influence  - per-vertex: up to 4 joint indices + 4 weights (24 B, GPU-
//                mirrored byte-for-byte by the SkinPass HLSL kernel).
//   Joint      - name + parent index + local bind transform + inverse bind.
//   Skeleton   - joints in TOPOLOGICAL order (every parent precedes its
//                children) so the global-transform walk is one linear pass.
//   Pose       - per-joint LOCAL (parent-relative) transforms. Empty = bind.
//
// Pipeline per frame:  Pose --compute_global--> globals --compute_palette-->
// palette (global[j] * inverse_bind[j]) --skin_vertices--> deformed verts.
// At the bind pose the palette is identity, so an un-posed skinned mesh
// renders exactly like the rigid mesh it came from.
//
// Linear blend skinning (LBS): the four palette matrices are weight-blended
// into one, which then transforms the position (w=1) and normal (upper 3x3,
// re-normalised). The GPU kernel performs the same operations in the same
// order so CPU and GPU outputs agree to FP-contraction noise.
//
// NaN-defensive throughout: non-finite weights are treated as 0, a joint
// index past the palette contributes identity, an all-zero weight set
// leaves the vertex untouched.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/scene/math.hpp>

namespace cardinal::scene {
struct Vertex;   // scene.hpp — skin.hpp is included BY scene.hpp
}

namespace cardinal::scene::skin {

constexpr u32 kMaxInfluences = 4;

// Per-vertex skinning influence. Layout is GPU-facing: the HLSL kernel reads
// it as uint2 (4 × u16 joints, little-endian pairs) + float4 weights.
struct Influence {
    u16   joint [kMaxInfluences]{0, 0, 0, 0};
    float weight[kMaxInfluences]{1.0f, 0.0f, 0.0f, 0.0f};
};
static_assert(sizeof(Influence) == 24, "Influence must stay 24 B (GPU mirror)");

struct Joint {
    cardinal::string name;
    i32  parent{-1};                        // index into Skeleton::joints; -1 = root
    Mat4 local_bind  {Mat4::identity()};    // parent-relative transform at bind
    Mat4 inverse_bind{Mat4::identity()};    // model space -> joint space at bind
};

struct Skeleton {
    cardinal::vector<Joint> joints;

    u32  joint_count() const noexcept { return static_cast<u32>(joints.size()); }
    // Index of the first joint with this name, or -1.
    i32  find(const cardinal::string& name) const noexcept;
    // True when every parent index is -1 or a LOWER in-range index (the
    // topological invariant compute_global relies on) and the joint count
    // fits the u16 Influence index.
    bool validate() const noexcept;
    // Recompute every inverse_bind from the local_bind hierarchy. For sources
    // that ship no inverse bind matrices (FBX clusters, hand-built rigs).
    void derive_inverse_bind();
};

// Per-joint LOCAL transforms. Empty (or shorter than the skeleton) means
// "bind pose" for the missing joints.
struct Pose {
    cardinal::vector<Mat4> local;
};

// Fill `out` with the skeleton's bind pose (local_bind per joint).
void bind_pose(const Skeleton& skel, Pose& out);

// global[j] = global[parent(j)] * local[j]. `out_global` is resized to the
// joint count. Joints missing from the pose use their local_bind.
void compute_global(const Skeleton& skel, const Pose& pose,
                    cardinal::vector<Mat4>& out_global);

// palette[j] = global[j] * inverse_bind[j]. `out_palette` is resized to the
// joint count. Identity at the bind pose.
void compute_palette(const Skeleton& skel, const cardinal::vector<Mat4>& global,
                     cardinal::vector<Mat4>& out_palette);

// Convenience: pose -> palette in one call. `scratch_global` is caller-owned
// so the per-frame path allocates nothing once warm.
void compute_palette(const Skeleton& skel, const Pose& pose,
                     cardinal::vector<Mat4>& scratch_global,
                     cardinal::vector<Mat4>& out_palette);

// Sanitise an influence: non-finite / negative weights -> 0, weights
// renormalised to sum 1 (an all-zero set becomes {1,0,0,0} on joint 0 —
// rigid-to-root, never a vanishing vertex).
Influence normalized(const Influence& in) noexcept;

// Linear-blend-skin ONE point. `palette_count` guards joint indices: an
// out-of-range joint contributes identity for its weight. The normal is
// transformed by the blended upper 3x3 and re-normalised (a zero-length
// result is passed through unchanged).
void skin_point(const Influence& infl,
                const Mat4* palette, u32 palette_count,
                const Vec3& in_pos, const Vec3& in_nrm,
                Vec3& out_pos, Vec3& out_nrm) noexcept;

// Skin `count` vertices (position + normal deformed; colour + UV copied).
// `out` may alias `in`. Null palette or zero palette_count copies verbatim.
void skin_vertices(const Vertex* in, const Influence* infl, u32 count,
                   const Mat4* palette, u32 palette_count,
                   Vertex* out) noexcept;

}  // namespace cardinal::scene::skin
