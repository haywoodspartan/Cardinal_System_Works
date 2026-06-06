#pragma once

// =============================================================================
// Cardinal Core — canonical math types.
//
// SINGLE SOURCE OF TRUTH for Vec2 / Vec3 / Vec4 / Mat3 / Mat4 / Quat across
// the whole engine. Modules that previously rolled their own (scene::Vec3,
// physics::Vec3, world::Vec3, physics::Quat, scene::Mat4, physics::Mat3,
// …) now alias these via `using` so memory layouts AND semantics are
// guaranteed identical between modules — no more `static_cast<scene::Vec3>(
// physics::Vec3{...})`-style boilerplate at boundaries.
//
// Layout invariants (CRITICAL — relied on by RHI uniform-buffer uploads,
// GPU mesh vertex layouts, and cross-module type-punning):
//
//   Vec2  : { f32 x, y }                   — 8 bytes, std140-friendly when packed
//   Vec3  : { f32 x, y, z }                — 12 bytes (NOT padded to 16)
//   Vec4  : { f32 x, y, z, w }             — 16 bytes
//   Mat3  : f32 m[3][3]  (row-major)       — 36 bytes
//   Mat4  : f32 m[4][4]  (column-major)    — 64 bytes; HLSL/SPIR-V on upload
//   Quat  : { f32 x, y, z, w }             — 16 bytes (xyzw order)
//
// All operators are constexpr where possible; free functions live outside
// the type bodies for ADL-friendly use (`dot(a, b)` not `a.dot(b)`).
// =============================================================================

#include <cardinal/core/types.hpp>

#include <cmath>

namespace cardinal::core {

// =============================================================================
// Vectors
// =============================================================================
struct Vec2 {
    f32 x{0}, y{0};
    constexpr Vec2() = default;
    constexpr Vec2(f32 xx, f32 yy) noexcept : x(xx), y(yy) {}
    constexpr Vec2 operator+(const Vec2& o) const noexcept { return { x+o.x, y+o.y }; }
    constexpr Vec2 operator-(const Vec2& o) const noexcept { return { x-o.x, y-o.y }; }
    constexpr Vec2 operator*(f32 s)         const noexcept { return { x*s,   y*s   }; }
    constexpr Vec2 operator-()              const noexcept { return { -x, -y }; }
    Vec2& operator+=(const Vec2& o) noexcept { x+=o.x; y+=o.y; return *this; }
    Vec2& operator-=(const Vec2& o) noexcept { x-=o.x; y-=o.y; return *this; }
    Vec2& operator*=(f32 s)         noexcept { x*=s;   y*=s;   return *this; }
};

struct Vec3 {
    f32 x{0}, y{0}, z{0};
    constexpr Vec3() = default;
    constexpr Vec3(f32 xx, f32 yy, f32 zz) noexcept : x(xx), y(yy), z(zz) {}
    constexpr Vec3 operator+(const Vec3& o) const noexcept { return { x+o.x, y+o.y, z+o.z }; }
    constexpr Vec3 operator-(const Vec3& o) const noexcept { return { x-o.x, y-o.y, z-o.z }; }
    constexpr Vec3 operator*(f32 s)         const noexcept { return { x*s,   y*s,   z*s   }; }
    constexpr Vec3 operator-()              const noexcept { return { -x, -y, -z }; }
    Vec3& operator+=(const Vec3& o) noexcept { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o) noexcept { x-=o.x; y-=o.y; z-=o.z; return *this; }
    Vec3& operator*=(f32 s)         noexcept { x*=s;   y*=s;   z*=s;   return *this; }
};

struct Vec4 {
    f32 x{0}, y{0}, z{0}, w{0};
    constexpr Vec4() = default;
    constexpr Vec4(f32 xx, f32 yy, f32 zz, f32 ww) noexcept : x(xx), y(yy), z(zz), w(ww) {}
};

// ----- Free vector functions ----------------------------------------------
inline constexpr f32 dot(const Vec2& a, const Vec2& b) noexcept { return a.x*b.x + a.y*b.y; }
inline constexpr f32 dot(const Vec3& a, const Vec3& b) noexcept { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline constexpr f32 dot(const Vec4& a, const Vec4& b) noexcept { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

inline Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}

inline f32  length(const Vec2& v) noexcept { return std::sqrt(dot(v, v)); }
inline f32  length(const Vec3& v) noexcept { return std::sqrt(dot(v, v)); }
inline f32  length(const Vec4& v) noexcept { return std::sqrt(dot(v, v)); }

inline Vec3 normalize(const Vec3& v) noexcept {
    const f32 l = length(v);
    return l > 1e-6f ? Vec3{ v.x/l, v.y/l, v.z/l } : Vec3{};
}
inline Vec2 normalize(const Vec2& v) noexcept {
    const f32 l = length(v);
    return l > 1e-6f ? Vec2{ v.x/l, v.y/l } : Vec2{};
}

inline Vec3 lerp(const Vec3& a, const Vec3& b, f32 t) noexcept {
    return { a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t, a.z + (b.z-a.z)*t };
}

// =============================================================================
// Quaternion (xyzw)
// =============================================================================
struct Quat {
    f32 x{0}, y{0}, z{0}, w{1};
    static constexpr Quat identity() noexcept { return {0, 0, 0, 1}; }
    Quat operator*(const Quat& o) const noexcept {
        return {
            w*o.x + x*o.w + y*o.z - z*o.y,
            w*o.y - x*o.z + y*o.w + z*o.x,
            w*o.z + x*o.y - y*o.x + z*o.w,
            w*o.w - x*o.x - y*o.y - z*o.z
        };
    }
};

inline Quat normalize(const Quat& q) noexcept {
    const f32 l2 = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    if (l2 <= 1e-12f) return Quat::identity();
    const f32 inv = 1.0f / std::sqrt(l2);
    return { q.x*inv, q.y*inv, q.z*inv, q.w*inv };
}

inline Vec3 rotate(const Quat& q, const Vec3& v) noexcept {
    // q*v*q^-1 expanded — about 15 mul + 15 add.
    const Vec3 u{ q.x, q.y, q.z };
    const f32  s = q.w;
    const Vec3 t = cross(u, v) * 2.0f;
    return v + t * s + cross(u, t);
}

inline Quat axis_angle(const Vec3& axis, f32 angle_rad) noexcept {
    const f32 h = angle_rad * 0.5f;
    const f32 s = std::sin(h);
    return { axis.x * s, axis.y * s, axis.z * s, std::cos(h) };
}

// Euler (XYZ, radians) ⇄ quaternion. Single canonical implementation —
// the editor gizmo + samples previously hand-rolled their own copies.
// q = qx ⊗ qy ⊗ qz; quat_to_euler_xyz is its exact inverse so the
// round-trip is stable for the usual range (pitch clamped at the pole).
inline Quat quat_from_euler_xyz(const Vec3& e) noexcept {
    const f32 hx = e.x * 0.5f, hy = e.y * 0.5f, hz = e.z * 0.5f;
    const f32 cx = std::cos(hx), sx = std::sin(hx);
    const f32 cy = std::cos(hy), sy = std::sin(hy);
    const f32 cz = std::cos(hz), sz = std::sin(hz);
    return { sx*cy*cz - cx*sy*sz,
             cx*sy*cz + sx*cy*sz,
             cx*cy*sz - sx*sy*cz,
             cx*cy*cz + sx*sy*sz };
}
inline Vec3 quat_to_euler_xyz(const Quat& q) noexcept {
    const f32 sinr = 2.0f * (q.w*q.x + q.y*q.z);
    const f32 cosr = 1.0f - 2.0f * (q.x*q.x + q.y*q.y);
    const f32 ex = std::atan2(sinr, cosr);
    f32 sp = 2.0f * (q.w*q.y - q.z*q.x);
    sp = (sp >  1.0f) ?  1.0f : ((sp < -1.0f) ? -1.0f : sp);
    const f32 ey = std::asin(sp);
    const f32 siny = 2.0f * (q.w*q.z + q.x*q.y);
    const f32 cosy = 1.0f - 2.0f * (q.y*q.y + q.z*q.z);
    const f32 ez = std::atan2(siny, cosy);
    return { ex, ey, ez };
}

// Integrate orientation by an angular velocity over dt. Re-normalised.
inline Quat integrate(const Quat& q, const Vec3& w, f32 dt) noexcept {
    const Quat wq{ w.x, w.y, w.z, 0.0f };
    const Quat dq = wq * q;
    const Quat r{ q.x + dq.x * 0.5f * dt,
                  q.y + dq.y * 0.5f * dt,
                  q.z + dq.z * 0.5f * dt,
                  q.w + dq.w * 0.5f * dt };
    return normalize(r);
}

// =============================================================================
// 3×3 matrix (row-major). Used for inertia tensors + quaternion-to-matrix.
// =============================================================================
struct Mat3 {
    f32 m[3][3]{};
    static constexpr Mat3 identity() noexcept {
        Mat3 r{};
        r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.0f;
        return r;
    }
    static constexpr Mat3 zero() noexcept { return Mat3{}; }
    static Mat3 diagonal(f32 a, f32 b, f32 c) noexcept {
        Mat3 r{};
        r.m[0][0] = a; r.m[1][1] = b; r.m[2][2] = c;
        return r;
    }
    Vec3 operator*(const Vec3& v) const noexcept {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z,
        };
    }
    Mat3 operator*(const Mat3& o) const noexcept {
        Mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][0]*o.m[0][j] + m[i][1]*o.m[1][j] + m[i][2]*o.m[2][j];
        return r;
    }
    Mat3 transposed() const noexcept {
        Mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = m[j][i];
        return r;
    }
};

inline Mat3 quat_to_mat3(const Quat& q) noexcept {
    const f32 xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
    const f32 xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
    const f32 wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
    Mat3 r{};
    r.m[0][0] = 1.0f - 2.0f*(yy + zz);
    r.m[0][1] =        2.0f*(xy - wz);
    r.m[0][2] =        2.0f*(xz + wy);
    r.m[1][0] =        2.0f*(xy + wz);
    r.m[1][1] = 1.0f - 2.0f*(xx + zz);
    r.m[1][2] =        2.0f*(yz - wx);
    r.m[2][0] =        2.0f*(xz - wy);
    r.m[2][1] =        2.0f*(yz + wx);
    r.m[2][2] = 1.0f - 2.0f*(xx + yy);
    return r;
}

// =============================================================================
// 4×4 matrix (column-major; m[c][r] indexing). Direct upload to HLSL/SPIR-V
// uniform buffers without transposing. Used for camera + transforms.
// =============================================================================
struct Mat4 {
    f32 m[4][4]{};

    static Mat4 identity() noexcept {
        Mat4 r{};
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
        return r;
    }

    static Mat4 translation(const Vec3& t) noexcept {
        Mat4 r = identity();
        r.m[3][0] = t.x; r.m[3][1] = t.y; r.m[3][2] = t.z;
        return r;
    }

    static Mat4 scaling(const Vec3& s) noexcept {
        Mat4 r{};
        r.m[0][0] = s.x; r.m[1][1] = s.y; r.m[2][2] = s.z; r.m[3][3] = 1.0f;
        return r;
    }

    // Euler XYZ rotation (radians, intrinsic order).
    static Mat4 rotation_xyz(const Vec3& r_rad) noexcept {
        const f32 cx = std::cos(r_rad.x), sx = std::sin(r_rad.x);
        const f32 cy = std::cos(r_rad.y), sy = std::sin(r_rad.y);
        const f32 cz = std::cos(r_rad.z), sz = std::sin(r_rad.z);
        Mat4 r = identity();
        r.m[0][0] =  cy*cz;
        r.m[0][1] =  cy*sz;
        r.m[0][2] = -sy;
        r.m[1][0] =  sx*sy*cz - cx*sz;
        r.m[1][1] =  sx*sy*sz + cx*cz;
        r.m[1][2] =  sx*cy;
        r.m[2][0] =  cx*sy*cz + sx*sz;
        r.m[2][1] =  cx*sy*sz - sx*cz;
        r.m[2][2] =  cx*cy;
        return r;
    }

    // Right-handed perspective, depth 0..1 (D3D / Vulkan after maintenance flip).
    static Mat4 perspective(f32 fov_y_rad, f32 aspect, f32 z_near, f32 z_far) noexcept {
        const f32 f = 1.0f / std::tan(fov_y_rad * 0.5f);
        Mat4 r{};
        r.m[0][0] = f / aspect;
        r.m[1][1] = f;
        r.m[2][2] = z_far / (z_near - z_far);
        r.m[2][3] = -1.0f;
        r.m[3][2] = (z_near * z_far) / (z_near - z_far);
        return r;
    }

    // Right-handed orthographic projection, depth mapped to z ∈ [0, 1]
    // (D3D/Vulkan convention — matches perspective() above). Column-major
    // m[col][row]. For a symmetric box (l=-h, r=h, b=-h, t=h) this reduces to
    // m[0][0]=1/h, m[1][1]=1/h, m[2][2]=-1/(f-n), m[3][2]=-n/(f-n) — the form
    // the directional shadow box was hand-rolling, so single-box behaviour is
    // bit-identical. Used for directional shadow-map light projections.
    static Mat4 ortho(f32 l, f32 r, f32 b, f32 t, f32 n, f32 f) noexcept {
        Mat4 o{};
        o.m[0][0] =  2.0f / (r - l);
        o.m[1][1] =  2.0f / (t - b);
        o.m[2][2] = -1.0f / (f - n);
        o.m[3][0] = -(r + l) / (r - l);
        o.m[3][1] = -(t + b) / (t - b);
        o.m[3][2] = -n / (f - n);
        o.m[3][3] =  1.0f;
        return o;
    }

    static Mat4 look_at(const Vec3& eye, const Vec3& target, const Vec3& up) noexcept {
        const Vec3 f = normalize(target - eye);
        const Vec3 s = normalize(cross(f, up));
        const Vec3 u = cross(s, f);
        Mat4 r = identity();
        r.m[0][0] =  s.x; r.m[1][0] =  s.y; r.m[2][0] =  s.z;
        r.m[0][1] =  u.x; r.m[1][1] =  u.y; r.m[2][1] =  u.z;
        r.m[0][2] = -f.x; r.m[1][2] = -f.y; r.m[2][2] = -f.z;
        r.m[3][0] = -dot(s, eye);
        r.m[3][1] = -dot(u, eye);
        r.m[3][2] =  dot(f, eye);
        return r;
    }

    Mat4 operator*(const Mat4& o) const noexcept {
        Mat4 r{};
        for (int c = 0; c < 4; ++c)
            for (int q = 0; q < 4; ++q) {
                f32 s = 0.0f;
                for (int k = 0; k < 4; ++k) s += m[k][q] * o.m[c][k];
                r.m[c][q] = s;
            }
        return r;
    }

    Vec4 operator*(const Vec4& v) const noexcept {
        Vec4 r{};
        r.x = m[0][0]*v.x + m[1][0]*v.y + m[2][0]*v.z + m[3][0]*v.w;
        r.y = m[0][1]*v.x + m[1][1]*v.y + m[2][1]*v.z + m[3][1]*v.w;
        r.z = m[0][2]*v.x + m[1][2]*v.y + m[2][2]*v.z + m[3][2]*v.w;
        r.w = m[0][3]*v.x + m[1][3]*v.y + m[2][3]*v.z + m[3][3]*v.w;
        return r;
    }

    // General 4x4 inverse via cofactor expansion. Returns identity when the
    // matrix is singular (det ~= 0).
    Mat4 inverse() const noexcept {
        const f32* a = &m[0][0];
        Mat4 inv{};
        f32* o = &inv.m[0][0];

        o[0]  =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
        o[1]  = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
        o[2]  =  a[1]*a[6]*a[15]  - a[1]*a[7]*a[14]  - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7]  - a[13]*a[3]*a[6];
        o[3]  = -a[1]*a[6]*a[11]  + a[1]*a[7]*a[10]  + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7]   + a[9]*a[3]*a[6];

        o[4]  = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
        o[5]  =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
        o[6]  = -a[0]*a[6]*a[15]  + a[0]*a[7]*a[14]  + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7]  + a[12]*a[3]*a[6];
        o[7]  =  a[0]*a[6]*a[11]  - a[0]*a[7]*a[10]  - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7]   - a[8]*a[3]*a[6];

        o[8]  =  a[4]*a[9]*a[15]  - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
        o[9]  = -a[0]*a[9]*a[15]  + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
        o[10] =  a[0]*a[5]*a[15]  - a[0]*a[7]*a[13]  - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7]  - a[12]*a[3]*a[5];
        o[11] = -a[0]*a[5]*a[11]  + a[0]*a[7]*a[9]   + a[4]*a[1]*a[11] - a[4]*a[3]*a[9]  - a[8]*a[1]*a[7]   + a[8]*a[3]*a[5];

        o[12] = -a[4]*a[9]*a[14]  + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
        o[13] =  a[0]*a[9]*a[14]  - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
        o[14] = -a[0]*a[5]*a[14]  + a[0]*a[6]*a[13]  + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6]  + a[12]*a[2]*a[5];
        o[15] =  a[0]*a[5]*a[10]  - a[0]*a[6]*a[9]   - a[4]*a[1]*a[10] + a[4]*a[2]*a[9]  + a[8]*a[1]*a[6]   - a[8]*a[2]*a[5];

        const f32 det = a[0]*o[0] + a[1]*o[4] + a[2]*o[8] + a[3]*o[12];
        if (std::abs(det) < 1e-8f) return Mat4::identity();
        const f32 inv_det = 1.0f / det;
        for (int i = 0; i < 16; ++i) o[i] *= inv_det;
        return inv;
    }
};

}  // namespace cardinal::core

// Signals to <cardinal/core/typedefs.hpp> that Vec3 / Vec4 / Mat4 are in
// scope, so it can expose Vec3Array / Vec4Array / Mat4Array aliases.
// Defined unconditionally here — the typedefs header reads it; if a
// caller includes typedefs.hpp WITHOUT math.hpp, only the primitive
// aliases (ByteVec, FloatVec, ...) appear.
#define CARDINAL_MATH_TYPES_VISIBLE 1
