// =============================================================================
// Cardinal — editor command bus implementation.
// =============================================================================
#include <cardinal/cmd/command.hpp>

#include <cardinal/scene/math.hpp>     // unproject_ndc_ray, ray_plane_y_intersect
#include <cardinal/level/level.hpp>    // AssetPlacement, PlaceResult
#include <cardinal/core/cmath.hpp>     // cardinal::round
#include <cardinal/core/algorithm.hpp> // cardinal::sort
#include <cardinal/core/utility.hpp>   // cardinal::move

namespace cardinal::cmd {

// ---- CommandRegistry --------------------------------------------------
void CommandRegistry::add(Command cmd) {
    commands_[cmd.id] = cardinal::move(cmd);
}
bool CommandRegistry::has(const cardinal::string& id) const {
    return commands_.find(id) != commands_.end();
}
const Command* CommandRegistry::find(const cardinal::string& id) const {
    auto it = commands_.find(id);
    return it == commands_.end() ? nullptr : &it->second;
}
CommandResult CommandRegistry::dispatch(const cardinal::string& id,
                                        CommandContext& ctx) const {
    const Command* c = find(id);
    if (c == nullptr || !c->run) return { false, "unknown command: " + id };
    return c->run(ctx);
}
cardinal::vector<cardinal::string> CommandRegistry::ids() const {
    cardinal::vector<cardinal::string> r;
    r.reserve(commands_.size());
    for (const auto& kv : commands_) r.push_back(kv.first);
    cardinal::sort(r.begin(), r.end());
    return r;
}

// ---- placement math (pure, testable, no Device) -----------------------
bool screen_to_ground(const ViewProj& vp, float ndc_x, float ndc_y,
                      float plane_y, bool snap, float snap_step,
                      cardinal::core::Vec3* out_hit) {
    if (!vp.valid) return false;
    const auto ray = cardinal::scene::unproject_ndc_ray(ndc_x, ndc_y,
                                                        vp.view, vp.proj);
    cardinal::scene::Vec3 hit{};
    if (!cardinal::scene::ray_plane_y_intersect(ray, plane_y, &hit)) return false;
    if (snap && snap_step > 0.0f) {
        hit.x = cardinal::round(hit.x / snap_step) * snap_step;
        hit.z = cardinal::round(hit.z / snap_step) * snap_step;
    }
    if (out_hit) *out_hit = hit;
    return true;
}

// ---- built-in commands ------------------------------------------------
void register_builtin_commands(CommandRegistry& reg) {
    // world.place_asset — spawn the active asset at the viewport click point,
    // unprojecting with the CAPTURED render matrices (ctx.viewport). This is
    // the single, correct placement path: no aspect / camera re-derivation, so
    // the asset lands under the cursor regardless of build config or RTT
    // resize state.
    Command place;
    place.id    = "world.place_asset";
    place.label = "Place Asset";
    place.run   = [](CommandContext& ctx) -> CommandResult {
        if (ctx.placement == nullptr)       return { false, "no AssetPlacement bound" };
        if (ctx.active_asset_id.empty())    return { false, "no active asset selected" };
        if (!ctx.viewport.valid)            return { false, "no captured viewport projection" };
        cardinal::core::Vec3 hit{};
        if (!screen_to_ground(ctx.viewport, ctx.ndc_x, ctx.ndc_y, 0.0f,
                              ctx.snap_enabled, ctx.snap_step, &hit))
            return { false, "ray missed the ground plane" };
        const auto pr = ctx.placement->place(ctx.active_asset_id.c_str(),
                                             ctx.device, hit);
        if (pr.actor == 0u) return { false, "placement produced no actor" };
        ctx.result_actor  = pr.actor;
        ctx.result_entity = pr.primary_entity;
        ctx.result_hit    = hit;
        return { true, {} };
    };
    reg.add(cardinal::move(place));
}

}  // namespace cardinal::cmd
