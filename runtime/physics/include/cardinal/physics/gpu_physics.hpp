#pragma once

// =============================================================================
// Cardinal — GPU physics passes.
//
// Three graph-hosted passes that move the per-step physics work — what
// the CPU does today inside cardinal::physics::World::step — into
// declared GPU compute kernels. Same convention as the render / lighting
// GPU passes: CPU reference under graph::CpuBackend (deterministic
// regression coverage) + HLSL source-string for the eventual RhiBackend
// compute dispatch.
//
//   * IntegrationPass    — semi-implicit Euler. One thread per body:
//                          vel += accel*dt; vel *= damping; pos += vel*dt.
//                          Embarrassingly parallel — biggest fixed cost
//                          on the CPU and the obvious first move to GPU.
//
//   * BroadphasePass     — O(N²) AABB-vs-AABB overlap pair generation.
//                          2D dispatch: one thread per (i, j) candidate
//                          pair. A real production broadphase is sweep-
//                          and-prune or grid-based; the O(N²) version
//                          here is the reference impl that the
//                          accelerated GPU broadphase will be validated
//                          against once it lands.
//
//   * ContactSolverPass  — single-iteration sequential-impulse contact
//                          solver. Reads contacts (pos, normal, depth,
//                          body-a, body-b) + per-body inverse mass, then
//                          updates the velocity buffer in place. The
//                          GPU shader uses atomic-add for the inout
//                          velocity write so multi-contact-per-body
//                          updates don't race; the CPU reference walks
//                          contacts serially and produces the same
//                          end-state.
//
// All passes are NaN-defended at every public read site.
// =============================================================================

#include <cardinal/render/graph.hpp>
#include <cardinal/core/types.hpp>

namespace cardinal::physics::gpu {

// ---------------------------------------------------------------------------
// IntegrationPass
//   in_state  (SoA, 10 floats per body):
//     pos.xyz (3N), vel.xyz (3N), accel.xyz (3N), inv_mass (1N)
//   out_state (same layout, updated)
// ---------------------------------------------------------------------------
class IntegrationPass {
public:
    struct State {
        cardinal::render::graph::ResourceHandle in_state;
        cardinal::render::graph::ResourceHandle out_state;
        cardinal::u32 body_count {0};
        float         dt         {1.0f / 60.0f};
        float         damping    {0.999f};
        // Stats:
        cardinal::u32 bodies_advanced {0};   // bodies with non-zero inv_mass
    };

    static cardinal::shared_ptr<State> add_to_graph(
        cardinal::render::graph::Graph& g,
        cardinal::render::graph::ResourceHandle in_state,
        cardinal::u32 body_count,
        float dt = 1.0f / 60.0f,
        float damping = 0.999f);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// BroadphasePass — O(N²) AABB pair generation
//   in_aabbs (SoA, 6N floats): min_x, min_y, min_z, max_x, max_y, max_z
// Outputs:
//   out_pairs (2 * max_pairs u32): pair list as (a_index, b_index)
//   out_count (1 u32): number of pairs actually produced
// ---------------------------------------------------------------------------
class BroadphasePass {
public:
    struct State {
        cardinal::render::graph::ResourceHandle in_aabbs;
        cardinal::render::graph::ResourceHandle out_pairs;
        cardinal::render::graph::ResourceHandle out_count;
        cardinal::u32 body_count {0};
        cardinal::u32 max_pairs  {0};
        // Stats:
        cardinal::u32 pairs_found {0};
        cardinal::u32 pairs_truncated {0};   // dropped because max_pairs exceeded
    };

    static cardinal::shared_ptr<State> add_to_graph(
        cardinal::render::graph::Graph& g,
        cardinal::render::graph::ResourceHandle in_aabbs,
        cardinal::u32 body_count,
        cardinal::u32 max_pairs);

    static const char* hlsl_source() noexcept;
};

// ---------------------------------------------------------------------------
// ContactSolverPass — one Gauss-Seidel iteration of sequential impulse.
//   in_contacts (10 floats × C): pos.xyz, normal.xyz, depth, _pad,
//                                u32 body_a (as float bits), u32 body_b
//   in_inv_mass (N floats):     per-body inverse mass (0 = static)
//   inout_vel   (3N floats SoA): per-body linear velocity
// ---------------------------------------------------------------------------
class ContactSolverPass {
public:
    struct State {
        cardinal::render::graph::ResourceHandle in_contacts;
        cardinal::render::graph::ResourceHandle in_inv_mass;
        cardinal::render::graph::ResourceHandle inout_vel;
        cardinal::u32 contact_count {0};
        cardinal::u32 body_count    {0};
        float         restitution   {0.0f};
        // Stats:
        cardinal::u32 contacts_resolved {0};
    };

    static cardinal::shared_ptr<State> add_to_graph(
        cardinal::render::graph::Graph& g,
        cardinal::render::graph::ResourceHandle in_contacts,
        cardinal::render::graph::ResourceHandle in_inv_mass,
        cardinal::render::graph::ResourceHandle inout_vel,
        cardinal::u32 contact_count,
        cardinal::u32 body_count,
        float restitution = 0.0f);

    static const char* hlsl_source() noexcept;
};

}  // namespace cardinal::physics::gpu
