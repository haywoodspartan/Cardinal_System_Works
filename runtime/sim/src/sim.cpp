#include <cardinal/sim/sim.hpp>

#include <cardinal/actor/component.hpp>
#include <cardinal/core/async.hpp>
#include <cardinal/core/cmath.hpp>       // cardinal::isfinite
#include <cardinal/core/log.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::remove_if/max
#include <cardinal/core/utility.hpp>     // cardinal::move
// cardinal::make_unique/vector/function arrive via sim.hpp

namespace cardinal::sim {

const char* tick_group_name(TickGroup g) noexcept {
    switch (g) {
        case TickGroup::PreUpdate:   return "PreUpdate";
        case TickGroup::PrePhysics:  return "PrePhysics";
        case TickGroup::Physics:     return "Physics";
        case TickGroup::PostPhysics: return "PostPhysics";
        case TickGroup::Update:      return "Update";
        case TickGroup::LateUpdate:  return "LateUpdate";
        case TickGroup::Render:      return "Render";
        case TickGroup::Count:       return "(count)";
    }
    return "?";
}

SimWorld::SimWorld(const SimDesc& desc)
    : desc_(desc), world_(cardinal::make_unique<cardinal::actor::World>())
{
    paused_ = desc.start_paused;
}
SimWorld::~SimWorld() = default;

void SimWorld::pause()  { paused_ = true; }
void SimWorld::resume() { paused_ = false; }
void SimWorld::step_one_frame() { single_step_ = true; paused_ = true; }
void SimWorld::set_time_scale(float s) {
    // The 3f3b05a fix guarded real_dt at the tick INGRESS, but missed
    // this SECOND ingress: scaled_dt = real_dt * time_scale_ at line
    // 134. A NaN time_scale_ poisons scaled_dt independently of how
    // clean real_dt is, then `physics_accum_ += NaN` makes the
    // accumulator NaN and `accum >= fixed_dt` stays false forever —
    // physics never substeps, every body frozen — exactly the bug
    // 3f3b05a was supposed to close. +Inf time_scale_ would make
    // scaled_dt +Inf and pin physics at max_substeps every frame.
    // The original `s < 0.0f` ordered compare let both through (NaN<0
    // false, +Inf<0 false). cardinal::isfinite covers all three
    // non-finite values; the negative-clamp intent is preserved.
    if (!cardinal::isfinite(s) || s < 0.0f) s = 0.0f;
    time_scale_ = s;
}
bool  SimWorld::paused()     const noexcept { return paused_; }
float SimWorld::time_scale() const noexcept { return time_scale_; }

SimWorld::HandlerId SimWorld::add_handler(TickGroup g, TickFn fn) {
    const HandlerId id = next_handler_id_++;
    handlers_[static_cast<usize>(g)].push_back(Sub{ id, cardinal::move(fn) });
    return id;
}
void SimWorld::remove_handler(HandlerId id) {
    for (auto& list : handlers_) {
        list.erase(cardinal::remove_if(list.begin(), list.end(),
            [id](const Sub& s){ return s.id == id; }), list.end());
    }
}

void SimWorld::run_group_(TickGroup g, float dt) {
    const auto& list = handlers_[static_cast<usize>(g)];
    if (list.empty()) return;
    cardinal::async::FrameScope _phase(tick_group_name(g));

    // Opt-in parallel dispatch — fan handlers out across workers when
    // SimDesc::parallel_handlers is true AND a JobSystem is available
    // AND we have at least two handlers to run (otherwise sequential
    // is identical and dispatch overhead is wasted).
    //
    // We build the task vector once per group call. Per-handler overhead
    // is one cardinal::function copy — negligible against the work each
    // handler does (every realistic handler touches dozens of entities).
    if (desc_.parallel_handlers && list.size() > 1
        && cardinal::async::pool() != nullptr)
    {
        cardinal::vector<cardinal::function<void()>> tasks;
        tasks.reserve(list.size());
        for (const auto& s : list) {
            if (!s.fn) continue;
            auto fn_copy = s.fn;
            tasks.emplace_back([fn_copy, dt]() { fn_copy(dt); });
        }
        cardinal::async::parallel_invoke(tasks);
        return;
    }

    // Sequential default — preserves per-handler ordering, no overhead.
    for (const auto& s : list) if (s.fn) s.fn(dt);
}

void SimWorld::integrate_physics_(float dt) {
    cardinal::async::FrameScope _phase(tick_group_name(TickGroup::Physics));
    constexpr float kGravity = -9.80665f;
    for (auto& aptr : world_->actors()) {
        if (!aptr->alive()) continue;
        auto* rb = aptr->get_component<cardinal::actor::RigidBodyComponent>();
        auto* tr = aptr->get_component<cardinal::actor::TransformComponent>();
        if (rb == nullptr || tr == nullptr) continue;
        if (rb->kinematic) continue;
        if (rb->use_gravity) rb->acceleration.y += kGravity;
        rb->velocity.x += rb->acceleration.x * dt;
        rb->velocity.y += rb->acceleration.y * dt;
        rb->velocity.z += rb->acceleration.z * dt;
        const float damp = cardinal::max(0.0f, 1.0f - rb->linear_damping * dt);
        rb->velocity.x *= damp;
        rb->velocity.y *= damp;
        rb->velocity.z *= damp;
        tr->translation.x += rb->velocity.x * dt;
        tr->translation.y += rb->velocity.y * dt;
        tr->translation.z += rb->velocity.z * dt;
        rb->acceleration = { 0, 0, 0 };
    }
}

float SimWorld::tick(float real_dt) {
    // `!(real_dt > 0.0f)` also catches NaN (every ordered compare with NaN
    // is false, so the old `real_dt < 0.0f` let NaN straight through). A
    // NaN real_dt would otherwise reach `physics_accum_ +=` and poison it
    // forever (NaN + x = NaN ⇒ `physics_accum_ >= fixed_dt` always false ⇒
    // fixed-step physics never substeps again, every body frozen) and turn
    // stats_.real_time_seconds permanently NaN. Treat non-finite as 0 —
    // identical to the existing negative→0 contract. +Inf is unaffected
    // here and still clamped by the max_real_dt guard below (unchanged).
    if (!(real_dt > 0.0f)) real_dt = 0.0f;
    if (real_dt > desc_.max_real_dt) real_dt = desc_.max_real_dt;

    stats_.real_time_seconds += real_dt;

    // When paused (and not single-stepping), only run Render. Editors still
    // want a redraw even when the sim is frozen.
    bool advance = !paused_;
    if (paused_ && single_step_) {
        advance      = true;
        single_step_ = false;
        stats_.single_step_armed = false;
    }

    const float scaled_dt = advance ? (real_dt * time_scale_) : 0.0f;
    if (advance) {
        run_group_(TickGroup::PreUpdate,   scaled_dt);
        run_group_(TickGroup::PrePhysics,  scaled_dt);

        // Fixed-step physics with sub-step cap.
        physics_accum_ += scaled_dt;
        u32 substeps = 0;
        while (physics_accum_ >= desc_.fixed_dt && substeps < desc_.max_substeps) {
            integrate_physics_(desc_.fixed_dt);
            physics_accum_ -= desc_.fixed_dt;
            ++substeps;
        }
        stats_.physics_substeps_last = substeps;

        run_group_(TickGroup::PostPhysics, scaled_dt);

        {
            cardinal::async::FrameScope _phase(tick_group_name(TickGroup::Update));
            world_->tick(scaled_dt);
        }
        run_group_(TickGroup::Update,      scaled_dt);
        run_group_(TickGroup::LateUpdate,  scaled_dt);

        world_->sweep();
        ++stats_.total_ticks;
        stats_.sim_time_seconds += scaled_dt;
    }

    // Render group always runs (so even paused sims draw).
    run_group_(TickGroup::Render, scaled_dt);

    stats_.time_scale = time_scale_;
    stats_.paused     = paused_;
    return scaled_dt;
}

SimStats SimWorld::stats() const noexcept { return stats_; }

}  // namespace cardinal::sim
