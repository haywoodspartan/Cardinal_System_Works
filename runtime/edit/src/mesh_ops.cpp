#include <cardinal/edit/mesh_ops.hpp>

#include <cardinal/scene/scene.hpp>

#include <cardinal/core/algorithm.hpp>    // cardinal::min/max/find
#include <cardinal/core/cmath.hpp>        // cardinal::sqrt/sin/cos/floor
#include <cardinal/core/containers.hpp>   // cardinal::unordered_map

namespace cardinal::edit::mesh_ops {

namespace {

constexpr float kPi = 3.14159265358979323846f;

inline Vec3 normalized(const Vec3& v) {
    const float l = cardinal::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if (l < 1e-8f) return Vec3{0,0,1};
    return Vec3{ v.x / l, v.y / l, v.z / l };
}
inline Vec3 v_cross(const Vec3& a, const Vec3& b) {
    return Vec3{ a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}
inline Vec3 add(const Vec3& a, const Vec3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 scl(const Vec3& a, float s)       { return {a.x*s, a.y*s, a.z*s};      }
inline Vec3 mul(const Vec3& a, const Vec3& b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
inline float v_dot(const Vec3& a, const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }

}  // namespace

// ---------------------------------------------------------------------------
// Primitive generators
// ---------------------------------------------------------------------------
cardinal::vector<Vertex> make_box(float size) {
    const float h = size * 0.5f;
    const Vec3 col{0.7f, 0.7f, 0.7f};
    auto face = [&](Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 n,
                    cardinal::vector<Vertex>& out) {
        out.push_back({p0, n, col}); out.push_back({p1, n, col}); out.push_back({p2, n, col});
        out.push_back({p0, n, col}); out.push_back({p2, n, col}); out.push_back({p3, n, col});
    };
    cardinal::vector<Vertex> v;
    v.reserve(36);
    face({-h,-h, h}, { h,-h, h}, { h, h, h}, {-h, h, h}, { 0, 0, 1}, v);  // +Z
    face({ h,-h,-h}, {-h,-h,-h}, {-h, h,-h}, { h, h,-h}, { 0, 0,-1}, v);  // -Z
    face({-h, h, h}, { h, h, h}, { h, h,-h}, {-h, h,-h}, { 0, 1, 0}, v);  // +Y
    face({-h,-h,-h}, { h,-h,-h}, { h,-h, h}, {-h,-h, h}, { 0,-1, 0}, v);  // -Y
    face({ h,-h, h}, { h,-h,-h}, { h, h,-h}, { h, h, h}, { 1, 0, 0}, v);  // +X
    face({-h,-h,-h}, {-h,-h, h}, {-h, h, h}, {-h, h,-h}, {-1, 0, 0}, v);  // -X
    return v;
}

cardinal::vector<Vertex> make_plane(float size, u32 subdivisions) {
    if (subdivisions < 1) subdivisions = 1;
    const u32 N = subdivisions;
    const float step = size / static_cast<float>(N);
    const Vec3 col{0.5f, 0.5f, 0.5f}, n{0,1,0};
    cardinal::vector<Vertex> v;
    v.reserve(N * N * 6);
    for (u32 j = 0; j < N; ++j) {
        for (u32 i = 0; i < N; ++i) {
            const float x0 = -size*0.5f + step*i;
            const float z0 = -size*0.5f + step*j;
            const float x1 = x0 + step;
            const float z1 = z0 + step;
            v.push_back({{x0,0,z0}, n, col}); v.push_back({{x1,0,z0}, n, col}); v.push_back({{x1,0,z1}, n, col});
            v.push_back({{x0,0,z0}, n, col}); v.push_back({{x1,0,z1}, n, col}); v.push_back({{x0,0,z1}, n, col});
        }
    }
    return v;
}

cardinal::vector<Vertex> make_sphere(float radius, u32 segments) {
    if (segments < 4) segments = 4;
    const u32 stacks = segments / 2;
    const Vec3 col{0.6f, 0.7f, 0.8f};
    cardinal::vector<Vertex> v;
    v.reserve(segments * stacks * 6);
    for (u32 j = 0; j < stacks; ++j) {
        const float p0 = (static_cast<float>(j  ) / stacks) * kPi;
        const float p1 = (static_cast<float>(j+1) / stacks) * kPi;
        for (u32 i = 0; i < segments; ++i) {
            const float t0 = (static_cast<float>(i  ) / segments) * 2.0f * kPi;
            const float t1 = (static_cast<float>(i+1) / segments) * 2.0f * kPi;
            auto pt = [&](float t, float p) {
                Vec3 nrm{ cardinal::sin(p)*cardinal::cos(t), cardinal::cos(p), cardinal::sin(p)*cardinal::sin(t) };
                return Vertex{ scl(nrm, radius), nrm, col };
            };
            Vertex a = pt(t0,p0), b = pt(t1,p0), c = pt(t1,p1), d = pt(t0,p1);
            v.push_back(a); v.push_back(b); v.push_back(c);
            v.push_back(a); v.push_back(c); v.push_back(d);
        }
    }
    return v;
}

cardinal::vector<Vertex> make_cylinder(float radius, float height, u32 segments) {
    if (segments < 3) segments = 3;
    const float h = height * 0.5f;
    const Vec3 col{0.7f, 0.6f, 0.5f};
    cardinal::vector<Vertex> v;
    v.reserve(segments * 12);
    for (u32 i = 0; i < segments; ++i) {
        const float t0 = (static_cast<float>(i  ) / segments) * 2.0f * kPi;
        const float t1 = (static_cast<float>(i+1) / segments) * 2.0f * kPi;
        const float c0 = cardinal::cos(t0), s0 = cardinal::sin(t0);
        const float c1 = cardinal::cos(t1), s1 = cardinal::sin(t1);
        Vec3 n0{ c0, 0, s0 }, n1{ c1, 0, s1 };
        Vec3 b0{ c0*radius,-h, s0*radius };
        Vec3 b1{ c1*radius,-h, s1*radius };
        Vec3 t0p{ c0*radius, h, s0*radius };
        Vec3 t1p{ c1*radius, h, s1*radius };
        // Side
        v.push_back({b0, n0, col}); v.push_back({b1, n1, col}); v.push_back({t1p, n1, col});
        v.push_back({b0, n0, col}); v.push_back({t1p, n1, col}); v.push_back({t0p, n0, col});
        // Top cap
        Vec3 up{0,1,0};
        v.push_back({{0, h, 0}, up, col}); v.push_back({t0p, up, col}); v.push_back({t1p, up, col});
        // Bottom cap
        Vec3 dn{0,-1,0};
        v.push_back({{0,-h, 0}, dn, col}); v.push_back({b1,  dn, col}); v.push_back({b0,  dn, col});
    }
    return v;
}

cardinal::vector<Vertex> make_cone(float radius, float height, u32 segments) {
    if (segments < 3) segments = 3;
    const float h = height * 0.5f;
    const Vec3 col{0.7f, 0.5f, 0.4f};
    cardinal::vector<Vertex> v;
    v.reserve(segments * 6);
    Vec3 apex{0, h, 0};
    for (u32 i = 0; i < segments; ++i) {
        const float t0 = (static_cast<float>(i  ) / segments) * 2.0f * kPi;
        const float t1 = (static_cast<float>(i+1) / segments) * 2.0f * kPi;
        Vec3 b0{ cardinal::cos(t0)*radius,-h, cardinal::sin(t0)*radius };
        Vec3 b1{ cardinal::cos(t1)*radius,-h, cardinal::sin(t1)*radius };
        Vec3 fn = normalized(v_cross(sub(b1, b0), sub(apex, b0)));
        v.push_back({b0, fn, col}); v.push_back({b1, fn, col}); v.push_back({apex, fn, col});
        // Bottom
        Vec3 dn{0,-1,0};
        v.push_back({{0,-h,0}, dn, col}); v.push_back({b1, dn, col}); v.push_back({b0, dn, col});
    }
    return v;
}

cardinal::vector<Vertex> make_torus(float major, float minor, u32 maj_segs, u32 min_segs) {
    if (maj_segs < 3) maj_segs = 3;
    if (min_segs < 3) min_segs = 3;
    const Vec3 col{0.5f, 0.7f, 0.6f};
    cardinal::vector<Vertex> v;
    v.reserve(maj_segs * min_segs * 6);
    auto pt = [&](float u, float vp) {
        const float ct = cardinal::cos(u),     st = cardinal::sin(u);
        const float cp = cardinal::cos(vp),    sp = cardinal::sin(vp);
        Vec3 pos{ (major + minor*cp) * ct, minor * sp, (major + minor*cp) * st };
        Vec3 nrm{ ct * cp, sp, st * cp };
        return Vertex{ pos, nrm, col };
    };
    for (u32 i = 0; i < maj_segs; ++i) {
        const float u0 = (static_cast<float>(i  ) / maj_segs) * 2.0f * kPi;
        const float u1 = (static_cast<float>(i+1) / maj_segs) * 2.0f * kPi;
        for (u32 j = 0; j < min_segs; ++j) {
            const float v0 = (static_cast<float>(j  ) / min_segs) * 2.0f * kPi;
            const float v1 = (static_cast<float>(j+1) / min_segs) * 2.0f * kPi;
            Vertex a=pt(u0,v0), b=pt(u1,v0), c=pt(u1,v1), d=pt(u0,v1);
            v.push_back(a); v.push_back(b); v.push_back(c);
            v.push_back(a); v.push_back(c); v.push_back(d);
        }
    }
    return v;
}

cardinal::vector<Vertex> make_disk(float radius, u32 segments) {
    if (segments < 3) segments = 3;
    const Vec3 col{0.6f, 0.6f, 0.6f}, n{0,1,0};
    cardinal::vector<Vertex> v;
    v.reserve(segments * 3);
    for (u32 i = 0; i < segments; ++i) {
        const float t0 = (static_cast<float>(i  ) / segments) * 2.0f * kPi;
        const float t1 = (static_cast<float>(i+1) / segments) * 2.0f * kPi;
        v.push_back({{0,0,0}, n, col});
        v.push_back({{ cardinal::cos(t0)*radius, 0, cardinal::sin(t0)*radius }, n, col});
        v.push_back({{ cardinal::cos(t1)*radius, 0, cardinal::sin(t1)*radius }, n, col});
    }
    return v;
}

// ---------------------------------------------------------------------------
// Editing ops
// ---------------------------------------------------------------------------
cardinal::vector<Vertex> subdivide(const cardinal::vector<Vertex>& in, u32 levels) {
    cardinal::vector<Vertex> cur = in;
    for (u32 lvl = 0; lvl < levels; ++lvl) {
        cardinal::vector<Vertex> out;
        out.reserve(cur.size() * 4);
        for (usize i = 0; i + 2 < cur.size(); i += 3) {
            const Vertex& a = cur[i+0];
            const Vertex& b = cur[i+1];
            const Vertex& c = cur[i+2];
            auto mid = [](const Vertex& x, const Vertex& y) {
                Vertex r{};
                r.position = scl(add(x.position, y.position), 0.5f);
                r.normal   = normalized(add(x.normal,   y.normal));
                r.color    = scl(add(x.color,    y.color),    0.5f);
                return r;
            };
            Vertex ab = mid(a,b), bc = mid(b,c), ca = mid(c,a);
            // a, ab, ca
            out.push_back(a); out.push_back(ab); out.push_back(ca);
            out.push_back(ab); out.push_back(b); out.push_back(bc);
            out.push_back(ca); out.push_back(bc); out.push_back(c);
            out.push_back(ab); out.push_back(bc); out.push_back(ca);
        }
        cur.swap(out);
    }
    return cur;
}

cardinal::vector<Vertex> mirror(const cardinal::vector<Vertex>& in, u32 axis) {
    cardinal::vector<Vertex> out;
    out.reserve(in.size());
    auto flip = [axis](Vec3 v) {
        if (axis == 0) v.x = -v.x;
        else if (axis == 1) v.y = -v.y;
        else                v.z = -v.z;
        return v;
    };
    for (usize i = 0; i + 2 < in.size(); i += 3) {
        Vertex a = in[i+0], b = in[i+1], c = in[i+2];
        a.position = flip(a.position); a.normal = flip(a.normal);
        b.position = flip(b.position); b.normal = flip(b.normal);
        c.position = flip(c.position); c.normal = flip(c.normal);
        // Reverse winding so the mirrored face still points outward.
        out.push_back(a); out.push_back(c); out.push_back(b);
    }
    return out;
}

cardinal::vector<Vertex> decimate_cluster(const cardinal::vector<Vertex>& in, float cell_size) {
    // `cell_size <= 0.0f` is NaN-blind (NaN <= 0 is unordered-false).
    // A NaN cell_size flowed to `p.x / NaN = NaN`, then `static_cast
    // <i32>(NaN)` is UNDEFINED BEHAVIOR per [conv.fpint]p1. Same
    // float→int UB class as tex_ops::fclamp_u8 (3bc8360). Reject non-
    // finite alongside the existing positive-clamp; preserves the
    // documented "cell<=0 → passthrough" semantic for sane inputs.
    if (!cardinal::isfinite(cell_size) || cell_size <= 0.0f) return in;
    struct Key { i32 x, y, z; bool operator==(const Key& o) const { return x==o.x && y==o.y && z==o.z; } };
    struct KeyHash { usize operator()(const Key& k) const noexcept {
        return cardinal::hash<i64>{}( static_cast<i64>(k.x) * 73856093ll
                               ^ static_cast<i64>(k.y) * 19349663ll
                               ^ static_cast<i64>(k.z) * 83492791ll); }};
    cardinal::unordered_map<Key, Vertex, KeyHash> bucket;
    auto key_of = [cell_size](const Vec3& p) {
        // Sanitize per-component: a NaN vertex coord (caller bug —
        // mesh importers sometimes hand back NaN positions on
        // degenerate input) would re-introduce the UB cast even with
        // cell_size finite. Map NaN → 0 (cell origin); a degenerate
        // bucket is preferable to UB.
        const float fx = cardinal::isfinite(p.x) ? p.x : 0.0f;
        const float fy = cardinal::isfinite(p.y) ? p.y : 0.0f;
        const float fz = cardinal::isfinite(p.z) ? p.z : 0.0f;
        return Key{ static_cast<i32>(cardinal::floor(fx / cell_size)),
                    static_cast<i32>(cardinal::floor(fy / cell_size)),
                    static_cast<i32>(cardinal::floor(fz / cell_size)) };
    };
    cardinal::vector<Vertex> remap_v;
    remap_v.reserve(in.size());
    cardinal::vector<Key> keys;
    keys.reserve(in.size());
    for (const auto& v : in) {
        Key k = key_of(v.position);
        keys.push_back(k);
        auto it = bucket.find(k);
        if (it == bucket.end()) { bucket.emplace(k, v); }
        else {
            // Average into existing rep.
            it->second.position = scl(add(it->second.position, v.position), 0.5f);
            it->second.normal   = normalized(add(it->second.normal, v.normal));
            it->second.color    = scl(add(it->second.color, v.color),    0.5f);
        }
    }
    cardinal::vector<Vertex> out;
    out.reserve(in.size());
    for (usize i = 0; i + 2 < in.size(); i += 3) {
        const Key& k0 = keys[i+0];
        const Key& k1 = keys[i+1];
        const Key& k2 = keys[i+2];
        if (k0 == k1 || k1 == k2 || k0 == k2) continue;
        out.push_back(bucket[k0]);
        out.push_back(bucket[k1]);
        out.push_back(bucket[k2]);
    }
    return out;
}

cardinal::vector<Vertex> smooth_laplacian(const cardinal::vector<Vertex>& in,
                                     u32 iterations, float lambda)
{
    // Build vertex equivalence by exact position match (good enough for our
    // primitive meshes that share verts at seams). For each rep, accumulate
    // its 1-ring neighbours, then move toward the centroid.
    if (in.empty()) return in;
    struct Key { float x, y, z; bool operator==(const Key& o) const noexcept { return x==o.x && y==o.y && z==o.z; } };
    struct KeyHash { usize operator()(const Key& k) const noexcept {
        usize h = cardinal::hash<float>{}(k.x);
        h ^= cardinal::hash<float>{}(k.y) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= cardinal::hash<float>{}(k.z) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h; }};
    cardinal::unordered_map<Key, u32, KeyHash> rep;
    cardinal::vector<u32> rep_of(in.size(), 0);
    for (usize i = 0; i < in.size(); ++i) {
        Key k{ in[i].position.x, in[i].position.y, in[i].position.z };
        auto [it, inserted] = rep.emplace(k, static_cast<u32>(rep.size()));
        rep_of[i] = it->second;
    }
    const u32 R = static_cast<u32>(rep.size());
    cardinal::vector<Vec3> pos(R, Vec3{0,0,0});
    for (usize i = 0; i < in.size(); ++i) pos[rep_of[i]] = in[i].position;

    cardinal::vector<cardinal::vector<u32>> nbrs(R);
    auto add_nb = [&](u32 a, u32 b) {
        if (a == b) return;
        auto& list = nbrs[a];
        if (cardinal::find(list.begin(), list.end(), b) == list.end()) list.push_back(b);
    };
    for (usize i = 0; i + 2 < in.size(); i += 3) {
        u32 a = rep_of[i+0], b = rep_of[i+1], c = rep_of[i+2];
        add_nb(a,b); add_nb(a,c); add_nb(b,a); add_nb(b,c); add_nb(c,a); add_nb(c,b);
    }
    cardinal::vector<Vec3> next = pos;
    for (u32 it = 0; it < iterations; ++it) {
        for (u32 r = 0; r < R; ++r) {
            if (nbrs[r].empty()) continue;
            Vec3 c{0,0,0};
            for (u32 n : nbrs[r]) c = add(c, pos[n]);
            c = scl(c, 1.0f / static_cast<float>(nbrs[r].size()));
            next[r] = add(pos[r], scl(sub(c, pos[r]), lambda));
        }
        pos.swap(next);
    }
    cardinal::vector<Vertex> out = in;
    for (usize i = 0; i < in.size(); ++i) out[i].position = pos[rep_of[i]];
    return recompute_normals(out, /*smooth=*/true);
}

cardinal::vector<Vertex> recompute_normals(const cardinal::vector<Vertex>& in, bool smooth) {
    cardinal::vector<Vertex> out = in;
    if (in.empty()) return out;
    if (!smooth) {
        for (usize i = 0; i + 2 < out.size(); i += 3) {
            const Vec3 e1 = sub(out[i+1].position, out[i+0].position);
            const Vec3 e2 = sub(out[i+2].position, out[i+0].position);
            const Vec3 n  = normalized(v_cross(e1, e2));
            out[i+0].normal = n; out[i+1].normal = n; out[i+2].normal = n;
        }
        return out;
    }
    // Smooth: accumulate face normals into per-vertex by position-equivalence.
    struct Key { float x, y, z; bool operator==(const Key& o) const noexcept { return x==o.x && y==o.y && z==o.z; } };
    struct KeyHash { usize operator()(const Key& k) const noexcept {
        usize h = cardinal::hash<float>{}(k.x);
        h ^= cardinal::hash<float>{}(k.y) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= cardinal::hash<float>{}(k.z) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h; }};
    cardinal::unordered_map<Key, Vec3, KeyHash> n_acc;
    for (usize i = 0; i + 2 < in.size(); i += 3) {
        const Vec3 e1 = sub(in[i+1].position, in[i+0].position);
        const Vec3 e2 = sub(in[i+2].position, in[i+0].position);
        const Vec3 fn = normalized(v_cross(e1, e2));
        for (u32 k = 0; k < 3; ++k) {
            Key key{ in[i+k].position.x, in[i+k].position.y, in[i+k].position.z };
            auto it = n_acc.find(key);
            if (it == n_acc.end()) n_acc.emplace(key, fn);
            else                   it->second = add(it->second, fn);
        }
    }
    for (auto& v : out) {
        Key key{ v.position.x, v.position.y, v.position.z };
        auto it = n_acc.find(key);
        if (it != n_acc.end()) v.normal = normalized(it->second);
    }
    return out;
}

cardinal::vector<Vertex> bake_transform(const cardinal::vector<Vertex>& in,
                                   const cardinal::scene::Mat4& xform)
{
    cardinal::vector<Vertex> out;
    out.reserve(in.size());
    for (const auto& v : in) {
        cardinal::scene::Vec4 p4{ v.position.x, v.position.y, v.position.z, 1.0f };
        cardinal::scene::Vec4 wp = xform * p4;
        Vertex nv = v;
        nv.position = Vec3{ wp.x, wp.y, wp.z };
        // Rotate the normal — we use the rotation part of xform (col 0..2,
        // row 0..2). Simplest correct path: transform direction, then renorm.
        cardinal::scene::Vec4 n4{ v.normal.x, v.normal.y, v.normal.z, 0.0f };
        cardinal::scene::Vec4 wn = xform * n4;
        nv.normal = normalized(Vec3{ wn.x, wn.y, wn.z });
        out.push_back(nv);
    }
    return out;
}

cardinal::vector<Vertex> tint(const cardinal::vector<Vertex>& in, const Vec3& t) {
    cardinal::vector<Vertex> out = in;
    for (auto& v : out) v.color = mul(v.color, t);
    return out;
}

cardinal::vector<Vertex> append(cardinal::vector<Vertex> a, const cardinal::vector<Vertex>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

MeshStats stats(const cardinal::vector<Vertex>& in) {
    MeshStats s{};
    s.vert_count = static_cast<u32>(in.size());
    s.tri_count  = static_cast<u32>(in.size() / 3);
    if (in.empty()) return s;
    s.aabb_min = s.aabb_max = in.front().position;
    for (const auto& v : in) {
        s.aabb_min.x = cardinal::min(s.aabb_min.x, v.position.x);
        s.aabb_min.y = cardinal::min(s.aabb_min.y, v.position.y);
        s.aabb_min.z = cardinal::min(s.aabb_min.z, v.position.z);
        s.aabb_max.x = cardinal::max(s.aabb_max.x, v.position.x);
        s.aabb_max.y = cardinal::max(s.aabb_max.y, v.position.y);
        s.aabb_max.z = cardinal::max(s.aabb_max.z, v.position.z);
    }
    for (usize i = 0; i + 2 < in.size(); i += 3) {
        const Vec3 e1 = sub(in[i+1].position, in[i+0].position);
        const Vec3 e2 = sub(in[i+2].position, in[i+0].position);
        s.surface_area += 0.5f *
            cardinal::sqrt(v_dot(v_cross(e1, e2), v_cross(e1, e2)));
    }
    return s;
}

}  // namespace cardinal::edit::mesh_ops
