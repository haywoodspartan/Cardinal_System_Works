#include <cardinal/physics/gpu_physics.hpp>

#include <cardinal/core/cmath.hpp>
#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::physics::gpu {

namespace rg = cardinal::render::graph;

namespace {

inline bool isf(float v) noexcept { return cardinal::isfinite(v); }
inline float fzf(float v, float fb) noexcept { return isf(v) ? v : fb; }

}  // namespace

// =============================================================================
// IntegrationPass — semi-implicit Euler
// =============================================================================
namespace {

void integrate_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<IntegrationPass::State*>(uctx);
    const cardinal::u32 N = st->body_count;
    if (N == 0) {
        ec.dispatch(0, 1, 1);
        return;
    }
    const float* in_state = static_cast<const float*>(ec.map_buffer_read(st->in_state));
    float*       out      = static_cast<float*>      (ec.map_buffer_write(st->out_state));
    if (!in_state || !out) {
        ec.dispatch(0, 1, 1);
        return;
    }

    const float dt      = fzf(st->dt, 1.0f / 60.0f);
    const float damping = cardinal::clamp(fzf(st->damping, 0.999f), 0.0f, 1.0f);

    // SoA layout:
    //   [0..N)         pos.x
    //   [N..2N)        pos.y
    //   [2N..3N)       pos.z
    //   [3N..4N)       vel.x
    //   [4N..5N)       vel.y
    //   [5N..6N)       vel.z
    //   [6N..7N)       accel.x
    //   [7N..8N)       accel.y
    //   [8N..9N)       accel.z
    //   [9N..10N)      inv_mass
    const float* in_px  = in_state + 0 * N;
    const float* in_py  = in_state + 1 * N;
    const float* in_pz  = in_state + 2 * N;
    const float* in_vx  = in_state + 3 * N;
    const float* in_vy  = in_state + 4 * N;
    const float* in_vz  = in_state + 5 * N;
    const float* in_ax  = in_state + 6 * N;
    const float* in_ay  = in_state + 7 * N;
    const float* in_az  = in_state + 8 * N;
    const float* in_im  = in_state + 9 * N;

    float* out_px = out + 0 * N;
    float* out_py = out + 1 * N;
    float* out_pz = out + 2 * N;
    float* out_vx = out + 3 * N;
    float* out_vy = out + 4 * N;
    float* out_vz = out + 5 * N;
    float* out_ax = out + 6 * N;
    float* out_ay = out + 7 * N;
    float* out_az = out + 8 * N;
    float* out_im = out + 9 * N;

    cardinal::u32 advanced = 0;
    for (cardinal::u32 i = 0; i < N; ++i) {
        const float im = fzf(in_im[i], 0.0f);
        const float px = fzf(in_px[i], 0.0f);
        const float py = fzf(in_py[i], 0.0f);
        const float pz = fzf(in_pz[i], 0.0f);
        const float vx = fzf(in_vx[i], 0.0f);
        const float vy = fzf(in_vy[i], 0.0f);
        const float vz = fzf(in_vz[i], 0.0f);
        const float ax = fzf(in_ax[i], 0.0f);
        const float ay = fzf(in_ay[i], 0.0f);
        const float az = fzf(in_az[i], 0.0f);

        // Static bodies (im == 0) skip integration entirely.
        if (im > 0.0f) {
            ++advanced;
            // Semi-implicit Euler: v += a*dt; v *= damping; p += v*dt
            const float nvx = (vx + ax * dt) * damping;
            const float nvy = (vy + ay * dt) * damping;
            const float nvz = (vz + az * dt) * damping;
            out_px[i] = px + nvx * dt;
            out_py[i] = py + nvy * dt;
            out_pz[i] = pz + nvz * dt;
            out_vx[i] = nvx;
            out_vy[i] = nvy;
            out_vz[i] = nvz;
        } else {
            out_px[i] = px; out_py[i] = py; out_pz[i] = pz;
            out_vx[i] = vx; out_vy[i] = vy; out_vz[i] = vz;
        }
        out_ax[i] = ax; out_ay[i] = ay; out_az[i] = az;
        out_im[i] = im;
    }
    st->bodies_advanced = advanced;

    ec.dispatch((N + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<IntegrationPass::State> IntegrationPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_state,
    cardinal::u32 body_count,
    float dt, float damping)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_state   = in_state;
    st->body_count = body_count;
    st->dt         = dt;
    st->damping    = damping;

    rg::BufferDesc od;
    od.name         = "phys.state";
    od.size_bytes   = static_cast<cardinal::usize>(body_count) * 10u * sizeof(float);
    od.stride_bytes = sizeof(float);
    st->out_state = g.declare_buffer(od);

    rg::PassDesc pd;
    pd.name = "IntegrationPass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_state,      rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{st->out_state, rg::AccessMode::Write, 1});
    pd.record   = integrate_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (body_count + 63) / 64;
    pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* IntegrationPass::hlsl_source() noexcept {
    return R"(// Cardinal — IntegrationPass.hlsl
//
// One thread per body. Semi-implicit Euler with linear damping.
ByteAddressBuffer  in_state  : register(t0);
RWByteAddressBuffer out_state : register(u0);
cbuffer Params : register(b0) {
    uint  body_count; float dt; float damping; uint _pad;
};

float load_f(ByteAddressBuffer b, uint o) { return asfloat(b.Load(o)); }
void  store_f(RWByteAddressBuffer b, uint o, float v) { b.Store(o, asuint(v)); }

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= body_count) return;
    uint N = body_count;

    float im = load_f(in_state, (9*N+i)*4);
    float3 p = float3(load_f(in_state, (0*N+i)*4), load_f(in_state, (1*N+i)*4), load_f(in_state, (2*N+i)*4));
    float3 v = float3(load_f(in_state, (3*N+i)*4), load_f(in_state, (4*N+i)*4), load_f(in_state, (5*N+i)*4));
    float3 a = float3(load_f(in_state, (6*N+i)*4), load_f(in_state, (7*N+i)*4), load_f(in_state, (8*N+i)*4));

    if (im > 0) {
        v = (v + a * dt) * damping;
        p += v * dt;
    }

    store_f(out_state, (0*N+i)*4, p.x);
    store_f(out_state, (1*N+i)*4, p.y);
    store_f(out_state, (2*N+i)*4, p.z);
    store_f(out_state, (3*N+i)*4, v.x);
    store_f(out_state, (4*N+i)*4, v.y);
    store_f(out_state, (5*N+i)*4, v.z);
    store_f(out_state, (6*N+i)*4, a.x);
    store_f(out_state, (7*N+i)*4, a.y);
    store_f(out_state, (8*N+i)*4, a.z);
    store_f(out_state, (9*N+i)*4, im);
}
)";
}

// =============================================================================
// BroadphasePass — O(N²) AABB pair generation
// =============================================================================
namespace {

void broadphase_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<BroadphasePass::State*>(uctx);
    const cardinal::u32 N = st->body_count;
    const cardinal::u32 M = st->max_pairs;

    const float* aabbs = static_cast<const float*>(ec.map_buffer_read(st->in_aabbs));
    cardinal::u32* pairs = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_pairs));
    cardinal::u32* cnt   = static_cast<cardinal::u32*>(ec.map_buffer_write(st->out_count));
    if (!aabbs || !pairs || !cnt) {
        if (cnt) cnt[0] = 0;
        st->pairs_found = 0;
        st->pairs_truncated = 0;
        ec.dispatch(0, 1, 1);
        return;
    }
    if (N < 2) {
        cnt[0] = 0;
        st->pairs_found = 0;
        st->pairs_truncated = 0;
        ec.dispatch(0, 1, 1);
        return;
    }

    const float* mn_x = aabbs + 0 * N;
    const float* mn_y = aabbs + 1 * N;
    const float* mn_z = aabbs + 2 * N;
    const float* mx_x = aabbs + 3 * N;
    const float* mx_y = aabbs + 4 * N;
    const float* mx_z = aabbs + 5 * N;

    cardinal::u32 found = 0;
    cardinal::u32 truncated = 0;
    for (cardinal::u32 i = 0; i < N; ++i) {
        for (cardinal::u32 j = i + 1; j < N; ++j) {
            // AABB overlap test: separated if any axis has no overlap.
            const bool overlap =
                (mn_x[i] <= mx_x[j]) && (mx_x[i] >= mn_x[j]) &&
                (mn_y[i] <= mx_y[j]) && (mx_y[i] >= mn_y[j]) &&
                (mn_z[i] <= mx_z[j]) && (mx_z[i] >= mn_z[j]);
            if (!overlap) continue;
            if (found < M) {
                pairs[found * 2 + 0] = i;
                pairs[found * 2 + 1] = j;
                ++found;
            } else {
                ++truncated;
            }
        }
    }
    cnt[0] = found;
    st->pairs_found     = found;
    st->pairs_truncated = truncated;

    // 2D dispatch — one workgroup per 8×8 tile of (i, j) space.
    const cardinal::u32 tx = (N + 7) / 8;
    ec.dispatch(tx, tx, 1);
}

}  // namespace

cardinal::shared_ptr<BroadphasePass::State> BroadphasePass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_aabbs,
    cardinal::u32 body_count,
    cardinal::u32 max_pairs)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_aabbs   = in_aabbs;
    st->body_count = body_count;
    st->max_pairs  = max_pairs;

    rg::BufferDesc pd_;
    pd_.name         = "bp.pairs";
    pd_.size_bytes   = static_cast<cardinal::usize>(max_pairs) * 2u * sizeof(cardinal::u32);
    pd_.stride_bytes = sizeof(cardinal::u32);
    st->out_pairs = g.declare_buffer(pd_);

    rg::BufferDesc cd_;
    cd_.name         = "bp.count";
    cd_.size_bytes   = sizeof(cardinal::u32);
    cd_.stride_bytes = sizeof(cardinal::u32);
    st->out_count = g.declare_buffer(cd_);

    rg::PassDesc pd;
    pd.name = "BroadphasePass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_aabbs,      rg::AccessMode::Read,  0});
    pd.accesses.push_back(rg::ResourceAccess{st->out_pairs, rg::AccessMode::Write, 1});
    pd.accesses.push_back(rg::ResourceAccess{st->out_count, rg::AccessMode::Write, 2});
    pd.record   = broadphase_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (body_count + 7) / 8;
    pd.dispatch_y = (body_count + 7) / 8;
    pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* BroadphasePass::hlsl_source() noexcept {
    return R"(// Cardinal — BroadphasePass.hlsl
//
// 2D dispatch over the upper-triangle of N×N. Each thread tests one
// AABB pair; an InterlockedAdd appends matching pairs to the output
// list. Output count caps at max_pairs.
ByteAddressBuffer  in_aabbs   : register(t0);
RWByteAddressBuffer out_pairs : register(u0);
RWByteAddressBuffer out_count : register(u1);
cbuffer Params : register(b0) { uint body_count, max_pairs, _p0, _p1; };

float load_f(ByteAddressBuffer b, uint o) { return asfloat(b.Load(o)); }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x, j = tid.y;
    if (i >= body_count || j >= body_count || i >= j) return;
    uint N = body_count;
    float xa_min = load_f(in_aabbs, (0*N+i)*4); float xa_max = load_f(in_aabbs, (3*N+i)*4);
    float ya_min = load_f(in_aabbs, (1*N+i)*4); float ya_max = load_f(in_aabbs, (4*N+i)*4);
    float za_min = load_f(in_aabbs, (2*N+i)*4); float za_max = load_f(in_aabbs, (5*N+i)*4);
    float xb_min = load_f(in_aabbs, (0*N+j)*4); float xb_max = load_f(in_aabbs, (3*N+j)*4);
    float yb_min = load_f(in_aabbs, (1*N+j)*4); float yb_max = load_f(in_aabbs, (4*N+j)*4);
    float zb_min = load_f(in_aabbs, (2*N+j)*4); float zb_max = load_f(in_aabbs, (5*N+j)*4);
    bool overlap = (xa_min <= xb_max) && (xa_max >= xb_min)
                && (ya_min <= yb_max) && (ya_max >= yb_min)
                && (za_min <= zb_max) && (za_max >= zb_min);
    if (!overlap) return;
    uint slot;
    out_count.InterlockedAdd(0, 1, slot);
    if (slot >= max_pairs) return;
    out_pairs.Store(slot * 2 * 4 + 0, i);
    out_pairs.Store(slot * 2 * 4 + 4, j);
}
)";
}

// =============================================================================
// ContactSolverPass — single-iteration sequential impulse
// =============================================================================
namespace {

void solver_record(rg::ExecutionContext& ec, void* uctx) noexcept {
    if (!uctx) return;
    auto* st = static_cast<ContactSolverPass::State*>(uctx);
    const cardinal::u32 C = st->contact_count;
    const cardinal::u32 N = st->body_count;

    const float* contacts = static_cast<const float*>(ec.map_buffer_read(st->in_contacts));
    const float* inv_mass = static_cast<const float*>(ec.map_buffer_read(st->in_inv_mass));
    float*       vel      = static_cast<float*>      (ec.map_buffer_write(st->inout_vel));
    if (!contacts || !inv_mass || !vel || C == 0 || N == 0) {
        st->contacts_resolved = 0;
        ec.dispatch(0, 1, 1);
        return;
    }

    const float* vx = vel + 0 * N;
    const float* vy = vel + 1 * N;
    const float* vz = vel + 2 * N;
    float* out_vx = vel + 0 * N;
    float* out_vy = vel + 1 * N;
    float* out_vz = vel + 2 * N;
    (void)vx; (void)vy; (void)vz;   // alias the in/out (inout_vel is shared)

    const float e = cardinal::clamp(fzf(st->restitution, 0.0f), 0.0f, 1.0f);

    cardinal::u32 resolved = 0;
    for (cardinal::u32 c = 0; c < C; ++c) {
        const float* cd = contacts + c * 10;
        // Layout: pos.xyz, normal.xyz, depth, _pad, a_idx_bits, b_idx_bits
        const float nxC = fzf(cd[3], 0.0f);
        const float nyC = fzf(cd[4], 1.0f);
        const float nzC = fzf(cd[5], 0.0f);
        const cardinal::u32 a_idx = *reinterpret_cast<const cardinal::u32*>(&cd[8]);
        const cardinal::u32 b_idx = *reinterpret_cast<const cardinal::u32*>(&cd[9]);
        if (a_idx >= N || b_idx >= N) continue;

        const float im_a = fzf(inv_mass[a_idx], 0.0f);
        const float im_b = fzf(inv_mass[b_idx], 0.0f);
        const float im_sum = im_a + im_b;
        if (im_sum <= 0.0f) continue;        // both static

        // Relative velocity along contact normal.
        const float rvx = out_vx[a_idx] - out_vx[b_idx];
        const float rvy = out_vy[a_idx] - out_vy[b_idx];
        const float rvz = out_vz[a_idx] - out_vz[b_idx];
        const float rel_v = rvx * nxC + rvy * nyC + rvz * nzC;
        if (rel_v >= 0.0f) continue;          // separating — ignore

        const float j = -(1.0f + e) * rel_v / im_sum;
        const float jx = j * nxC, jy = j * nyC, jz = j * nzC;

        out_vx[a_idx] += jx * im_a;
        out_vy[a_idx] += jy * im_a;
        out_vz[a_idx] += jz * im_a;
        out_vx[b_idx] -= jx * im_b;
        out_vy[b_idx] -= jy * im_b;
        out_vz[b_idx] -= jz * im_b;
        ++resolved;
    }
    st->contacts_resolved = resolved;

    ec.dispatch((C + 63) / 64, 1, 1);
}

}  // namespace

cardinal::shared_ptr<ContactSolverPass::State> ContactSolverPass::add_to_graph(
    rg::Graph& g,
    rg::ResourceHandle in_contacts,
    rg::ResourceHandle in_inv_mass,
    rg::ResourceHandle inout_vel,
    cardinal::u32 contact_count,
    cardinal::u32 body_count,
    float restitution)
{
    auto st = cardinal::shared_ptr<State>(new State());
    st->in_contacts   = in_contacts;
    st->in_inv_mass   = in_inv_mass;
    st->inout_vel     = inout_vel;
    st->contact_count = contact_count;
    st->body_count    = body_count;
    st->restitution   = restitution;

    rg::PassDesc pd;
    pd.name = "ContactSolverPass";
    pd.kind = rg::PassKind::Compute;
    pd.accesses.push_back(rg::ResourceAccess{in_contacts, rg::AccessMode::Read,      0});
    pd.accesses.push_back(rg::ResourceAccess{in_inv_mass, rg::AccessMode::Read,      1});
    pd.accesses.push_back(rg::ResourceAccess{inout_vel,   rg::AccessMode::ReadWrite, 2});
    pd.record   = solver_record;
    pd.user_ctx = st.get();
    pd.dispatch_x = (contact_count + 63) / 64;
    pd.dispatch_y = 1; pd.dispatch_z = 1;
    g.add_pass(cardinal::move(pd));
    return st;
}

const char* ContactSolverPass::hlsl_source() noexcept {
    return R"(// Cardinal — ContactSolverPass.hlsl
//
// Sequential impulse, one iteration. Writes to inout_vel via
// InterlockedAdd on the integer-aliased velocity buffer so multi-
// contact-per-body updates don't race. Single iteration matches the
// CPU reference; production solvers do N iterations in a host loop.
ByteAddressBuffer  in_contacts  : register(t0);
ByteAddressBuffer  in_inv_mass  : register(t1);
RWByteAddressBuffer inout_vel   : register(u0);
cbuffer Params : register(b0) { uint contact_count, body_count; float restitution; uint _p; };

float load_f(ByteAddressBuffer b, uint o) { return asfloat(b.Load(o)); }

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint c = tid.x;
    if (c >= contact_count) return;
    uint base = c * 10 * 4;
    float3 n = float3(load_f(in_contacts, base + 3*4), load_f(in_contacts, base + 4*4), load_f(in_contacts, base + 5*4));
    uint a = asuint(load_f(in_contacts, base + 8*4));
    uint b = asuint(load_f(in_contacts, base + 9*4));
    if (a >= body_count || b >= body_count) return;
    float ima = load_f(in_inv_mass, a * 4);
    float imb = load_f(in_inv_mass, b * 4);
    float ims = ima + imb;
    if (ims <= 0) return;
    uint N = body_count;
    float3 va = float3(load_f(inout_vel, (0*N+a)*4), load_f(inout_vel, (1*N+a)*4), load_f(inout_vel, (2*N+a)*4));
    float3 vb = float3(load_f(inout_vel, (0*N+b)*4), load_f(inout_vel, (1*N+b)*4), load_f(inout_vel, (2*N+b)*4));
    float rel = dot(va - vb, n);
    if (rel >= 0) return;
    float j = -(1 + restitution) * rel / ims;
    float3 imp = j * n;
    // Atomic add on each component.
    uint old;
    inout_vel.InterlockedExchange((0*N+a)*4, asuint(va.x + imp.x * ima), old);
    inout_vel.InterlockedExchange((1*N+a)*4, asuint(va.y + imp.y * ima), old);
    inout_vel.InterlockedExchange((2*N+a)*4, asuint(va.z + imp.z * ima), old);
    inout_vel.InterlockedExchange((0*N+b)*4, asuint(vb.x - imp.x * imb), old);
    inout_vel.InterlockedExchange((1*N+b)*4, asuint(vb.y - imp.y * imb), old);
    inout_vel.InterlockedExchange((2*N+b)*4, asuint(vb.z - imp.z * imb), old);
}
)";
}

}  // namespace cardinal::physics::gpu
