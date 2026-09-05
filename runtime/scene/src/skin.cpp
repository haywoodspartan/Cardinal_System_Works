// =============================================================================
// Cardinal — Skeletal skinning core (CPU reference implementation).
//
// See skin.hpp. This TU is the ground truth the GPU SkinPass kernel
// (render/gpu_skin.cpp) mirrors op-for-op; tests/skin pins both against
// each other.
// =============================================================================

#include <cardinal/scene/skin.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/core/std/cmath.hpp>

namespace cardinal::scene::skin {

namespace {

inline bool finite(float f) noexcept { return cardinal::isfinite(f); }

// Weighted accumulate: acc += w * m (all 16 cells).
inline void madd(Mat4& acc, const Mat4& m, float w) noexcept {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            acc.m[c][r] += w * m.m[c][r];
}

}  // namespace

// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------
i32 Skeleton::find(const cardinal::string& name) const noexcept {
    for (usize i = 0; i < joints.size(); ++i)
        if (joints[i].name == name) return static_cast<i32>(i);
    return -1;
}

bool Skeleton::validate() const noexcept {
    if (joints.size() > 65535u) return false;      // u16 Influence index
    for (usize i = 0; i < joints.size(); ++i) {
        const i32 p = joints[i].parent;
        if (p < -1) return false;
        if (p >= 0 && static_cast<usize>(p) >= i) return false;   // must precede
    }
    return true;
}

void Skeleton::derive_inverse_bind() {
    // globals in topological order, then invert.
    cardinal::vector<Mat4> global(joints.size());
    for (usize i = 0; i < joints.size(); ++i) {
        const i32 p = joints[i].parent;
        if (p >= 0 && static_cast<usize>(p) < i)
            global[i] = global[static_cast<usize>(p)] * joints[i].local_bind;
        else
            global[i] = joints[i].local_bind;
        joints[i].inverse_bind = global[i].inverse();
    }
}

// ---------------------------------------------------------------------------
// Pose -> palette
// ---------------------------------------------------------------------------
void bind_pose(const Skeleton& skel, Pose& out) {
    out.local.resize(skel.joints.size());
    for (usize i = 0; i < skel.joints.size(); ++i)
        out.local[i] = skel.joints[i].local_bind;
}

void compute_global(const Skeleton& skel, const Pose& pose,
                    cardinal::vector<Mat4>& out_global) {
    const usize n = skel.joints.size();
    out_global.resize(n);
    for (usize i = 0; i < n; ++i) {
        const Mat4& local = (i < pose.local.size()) ? pose.local[i]
                                                    : skel.joints[i].local_bind;
        const i32 p = skel.joints[i].parent;
        if (p >= 0 && static_cast<usize>(p) < i)
            out_global[i] = out_global[static_cast<usize>(p)] * local;
        else
            out_global[i] = local;   // root, or malformed parent -> treated as root
    }
}

void compute_palette(const Skeleton& skel, const cardinal::vector<Mat4>& global,
                     cardinal::vector<Mat4>& out_palette) {
    const usize n = skel.joints.size();
    out_palette.resize(n);
    for (usize i = 0; i < n; ++i) {
        if (i < global.size())
            out_palette[i] = global[i] * skel.joints[i].inverse_bind;
        else
            out_palette[i] = Mat4::identity();
    }
}

void compute_palette(const Skeleton& skel, const Pose& pose,
                     cardinal::vector<Mat4>& scratch_global,
                     cardinal::vector<Mat4>& out_palette) {
    compute_global(skel, pose, scratch_global);
    compute_palette(skel, scratch_global, out_palette);
}

// ---------------------------------------------------------------------------
// Vertex kernel
// ---------------------------------------------------------------------------
Influence normalized(const Influence& in) noexcept {
    Influence o = in;
    float sum = 0.0f;
    for (u32 k = 0; k < kMaxInfluences; ++k) {
        float w = o.weight[k];
        if (!finite(w) || w < 0.0f) w = 0.0f;
        o.weight[k] = w;
        sum += w;
    }
    if (sum <= 1e-8f) {
        o.joint[0] = in.joint[0];
        o.weight[0] = 1.0f; o.weight[1] = o.weight[2] = o.weight[3] = 0.0f;
        return o;
    }
    const float inv = 1.0f / sum;
    for (u32 k = 0; k < kMaxInfluences; ++k) o.weight[k] *= inv;
    return o;
}

void skin_point(const Influence& infl,
                const Mat4* palette, u32 palette_count,
                const Vec3& in_pos, const Vec3& in_nrm,
                Vec3& out_pos, Vec3& out_nrm) noexcept {
    // Blend the palette matrices. Weights are used AS AUTHORED (the asset
    // path normalises at import) so a deliberately under-weighted vertex
    // isn't silently re-scaled here; only non-finite weights are dropped.
    Mat4  M{};
    float total = 0.0f;
    for (u32 k = 0; k < kMaxInfluences; ++k) {
        float w = infl.weight[k];
        if (!finite(w) || w == 0.0f) continue;
        total += w;
        const u32 j = infl.joint[k];
        if (palette != nullptr && j < palette_count) madd(M, palette[j], w);
        else                                          madd(M, Mat4::identity(), w);
    }
    if (total == 0.0f) { out_pos = in_pos; out_nrm = in_nrm; return; }

    // Position (w = 1).
    out_pos = {
        M.m[0][0]*in_pos.x + M.m[1][0]*in_pos.y + M.m[2][0]*in_pos.z + M.m[3][0],
        M.m[0][1]*in_pos.x + M.m[1][1]*in_pos.y + M.m[2][1]*in_pos.z + M.m[3][1],
        M.m[0][2]*in_pos.x + M.m[1][2]*in_pos.y + M.m[2][2]*in_pos.z + M.m[3][2],
    };
    // Normal (upper 3x3, w = 0), re-normalised.
    Vec3 n{
        M.m[0][0]*in_nrm.x + M.m[1][0]*in_nrm.y + M.m[2][0]*in_nrm.z,
        M.m[0][1]*in_nrm.x + M.m[1][1]*in_nrm.y + M.m[2][1]*in_nrm.z,
        M.m[0][2]*in_nrm.x + M.m[1][2]*in_nrm.y + M.m[2][2]*in_nrm.z,
    };
    const float l2 = n.x*n.x + n.y*n.y + n.z*n.z;
    if (l2 > 1e-12f) {
        const float inv = 1.0f / cardinal::sqrt(l2);
        n.x *= inv; n.y *= inv; n.z *= inv;
        out_nrm = n;
    } else {
        out_nrm = in_nrm;
    }
}

void skin_vertices(const Vertex* in, const Influence* infl, u32 count,
                   const Mat4* palette, u32 palette_count,
                   Vertex* out) noexcept {
    if (in == nullptr || out == nullptr) return;
    if (infl == nullptr || palette == nullptr || palette_count == 0) {
        if (out != in)
            for (u32 i = 0; i < count; ++i) out[i] = in[i];
        return;
    }
    for (u32 i = 0; i < count; ++i) {
        Vertex v = in[i];      // copy first: out may alias in
        skin_point(infl[i], palette, palette_count,
                   in[i].position, in[i].normal, v.position, v.normal);
        out[i] = v;
    }
}

}  // namespace cardinal::scene::skin
