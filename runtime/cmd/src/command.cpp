// =============================================================================
// Cardinal — editor command bus implementation.
// =============================================================================
#include <cardinal/cmd/command.hpp>

#include <cardinal/scene/math.hpp>     // unproject_ndc_ray, ray_plane_y_intersect
#include <cardinal/scene/scene.hpp>    // Scene / Entity / Camera (camera.*)
#include <cardinal/level/level.hpp>    // AssetPlacement, PlaceResult
#include <cardinal/actor/world.hpp>      // World (actor/layout commands)
#include <cardinal/actor/validation.hpp> // validate_world / auto_fix_world
#include <cardinal/edit/undo.hpp>        // UndoStack (edit.undo / edit.redo)
#include <cardinal/core/cmath.hpp>     // cardinal::round
#include <cardinal/core/algorithm.hpp> // cardinal::sort
#include <cardinal/core/utility.hpp>   // cardinal::move

namespace cardinal::cmd {

// ---- CommandRegistry --------------------------------------------------
// Keyed by StringId(id) — a u64 hash — so lookups are an O(1) DenseMap probe.
// The public API stays string-based; each Command keeps its id string for
// ids() + diagnostics.
void CommandRegistry::add(Command cmd) {
    commands_[cardinal::StringId(cmd.id)] = cardinal::move(cmd);
}
bool CommandRegistry::has(const cardinal::string& id) const {
    return commands_.contains(cardinal::StringId(id));
}
const Command* CommandRegistry::find(const cardinal::string& id) const {
    return commands_.find(cardinal::StringId(id));   // const V* (nullptr if absent)
}
CommandResult CommandRegistry::dispatch(const cardinal::string& id,
                                        CommandContext& ctx) const {
    const Command* c = find(id);
    if (c == nullptr || !c->run) return { false, "unknown command: " + id };
    return c->run(ctx);
}
bool CommandRegistry::is_enabled(const cardinal::string& id,
                                 const CommandContext& ctx) const {
    const Command* c = find(id);
    if (c == nullptr) return false;
    return c->enabled ? c->enabled(ctx) : true;
}
cardinal::vector<cardinal::string> CommandRegistry::ids() const {
    cardinal::vector<cardinal::string> r;
    r.reserve(commands_.size());
    commands_.for_each([&](const cardinal::StringId&, const Command& c) {
        r.push_back(c.id);
    });
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
        // Two input modes: a viewport CLICK (unproject the captured matrices)
        // or a DIRECT position (menu spawn). One command, one place path.
        cardinal::core::Vec3 hit{};
        if (ctx.viewport.valid) {
            if (!screen_to_ground(ctx.viewport, ctx.ndc_x, ctx.ndc_y, 0.0f,
                                  ctx.snap_enabled, ctx.snap_step, &hit))
                return { false, "ray missed the ground plane" };
        } else {
            hit = ctx.place_position;
            // Honour grid snap in direct-position mode too, so a menu/palette
            // spawn snaps identically to a click spawn (same context fields).
            if (ctx.snap_enabled && ctx.snap_step > 0.0f) {
                hit.x = cardinal::round(hit.x / ctx.snap_step) * ctx.snap_step;
                hit.z = cardinal::round(hit.z / ctx.snap_step) * ctx.snap_step;
            }
        }
        const auto pr = ctx.placement->place(ctx.active_asset_id.c_str(),
                                             ctx.device, hit);
        if (pr.actor == 0u) return { false, "placement produced no actor" };
        ctx.result_actor  = pr.actor;
        ctx.result_entity = pr.primary_entity;
        ctx.result_hit    = hit;
        return { true, {} };
    };
    place.enabled = [](const CommandContext& ctx) {
        return ctx.placement != nullptr && !ctx.active_asset_id.empty();
    };
    reg.add(cardinal::move(place));

    // Selection-driven enablement predicate shared by actor.* / layout.*.
    const auto has_selection = [](const CommandContext& ctx) {
        return ctx.world != nullptr && !ctx.selection.empty();
    };

    // ---- actor / layout commands (pure World ops — headless-testable) ----
    // These act on ctx.world + ctx.selection, so the same operation the
    // Outliner/menu invokes is unit-testable without the GUI.
    {
        Command c;
        c.id = "actor.duplicate"; c.label = "Duplicate Selected";
        c.run = [](CommandContext& ctx) -> CommandResult {
            if (ctx.world == nullptr)     return { false, "no world" };
            if (ctx.selection.empty())    return { false, "empty selection" };
            cardinal::u32 n = 0, last = 0;
            for (const auto id : ctx.selection)
                if (auto* d = ctx.world->duplicate(id)) { last = d->id(); ++n; }
            ctx.result_count = n;
            ctx.result_actor = last;
            return n ? CommandResult{ true, {} }
                     : CommandResult{ false, "nothing duplicated" };
        };
        c.enabled = has_selection;
        reg.add(cardinal::move(c));
    }
    {
        Command c;
        c.id = "actor.delete"; c.label = "Delete Selected";
        c.run = [](CommandContext& ctx) -> CommandResult {
            if (ctx.world == nullptr) return { false, "no world" };
            ctx.result_count = ctx.world->bulk_destroy(ctx.selection);
            return ctx.result_count ? CommandResult{ true, {} }
                                    : CommandResult{ false, "nothing deleted" };
        };
        c.enabled = has_selection;
        reg.add(cardinal::move(c));
    }
    {
        Command c;
        c.id = "layout.align"; c.label = "Align Selected";
        c.run = [](CommandContext& ctx) -> CommandResult {
            if (ctx.world == nullptr) return { false, "no world" };
            using Ax = cardinal::actor::World::Axis;
            using Md = cardinal::actor::World::AlignMode;
            const Ax ax = (ctx.axis == 2) ? Ax::Z : (ctx.axis == 1) ? Ax::Y : Ax::X;
            const Md md = (ctx.align_mode == 0) ? Md::Min
                        : (ctx.align_mode == 2) ? Md::Max : Md::Center;
            ctx.result_count = ctx.world->align_actors(ctx.selection, ax, md);
            return { true, {} };
        };
        c.enabled = has_selection;
        reg.add(cardinal::move(c));
    }
    {
        Command c;
        c.id = "layout.distribute"; c.label = "Distribute Selected";
        c.run = [](CommandContext& ctx) -> CommandResult {
            if (ctx.world == nullptr) return { false, "no world" };
            using Ax = cardinal::actor::World::Axis;
            const Ax ax = (ctx.axis == 2) ? Ax::Z : (ctx.axis == 1) ? Ax::Y : Ax::X;
            ctx.result_count = ctx.world->distribute_actors(ctx.selection, ax);
            return { true, {} };
        };
        c.enabled = has_selection;
        reg.add(cardinal::move(c));
    }
    {
        Command c;
        c.id = "layout.snap"; c.label = "Snap Selected to Grid";
        c.run = [](CommandContext& ctx) -> CommandResult {
            if (ctx.world == nullptr) return { false, "no world" };
            ctx.result_count =
                ctx.world->snap_actors_to_grid(ctx.selection, ctx.grid_step);
            return { true, {} };
        };
        c.enabled = has_selection;
        reg.add(cardinal::move(c));
    }
    const auto has_world = [](const CommandContext& ctx) { return ctx.world != nullptr; };
    {
        Command c;
        c.id = "world.validate"; c.label = "Validate Scene";
        c.run = [](CommandContext& ctx) -> CommandResult {
            if (ctx.world == nullptr) return { false, "no world" };
            const auto issues = cardinal::actor::validate_world(*ctx.world);
            ctx.result_count = static_cast<cardinal::u32>(issues.size());
            return { true, {} };   // ok regardless; result_count = # issues
        };
        c.enabled = has_world;
        reg.add(cardinal::move(c));
    }
    {
        Command c;
        c.id = "world.autofix"; c.label = "Auto-Fix Scene Issues";
        c.run = [](CommandContext& ctx) -> CommandResult {
            if (ctx.world == nullptr) return { false, "no world" };
            ctx.result_count = cardinal::actor::auto_fix_world(*ctx.world);
            return { true, {} };   // result_count = # fixes applied
        };
        c.enabled = has_world;
        reg.add(cardinal::move(c));
    }
    // ---- edit.undo / edit.redo — one dispatch path for the scene undo
    // stack (toolbar buttons, Ctrl+Z, and the palette all resolve to these).
    {
        Command c;
        c.id = "edit.undo"; c.label = "Undo";
        c.run = [](CommandContext& ctx) -> CommandResult {
            if (ctx.scene_undo == nullptr) return { false, "no undo stack" };
            return ctx.scene_undo->undo() ? CommandResult{ true, {} }
                                          : CommandResult{ false, "nothing to undo" };
        };
        c.enabled = [](const CommandContext& ctx) {
            return ctx.scene_undo != nullptr && ctx.scene_undo->can_undo();
        };
        reg.add(cardinal::move(c));
    }
    {
        Command c;
        c.id = "edit.redo"; c.label = "Redo";
        c.run = [](CommandContext& ctx) -> CommandResult {
            if (ctx.scene_undo == nullptr) return { false, "no undo stack" };
            return ctx.scene_undo->redo() ? CommandResult{ true, {} }
                                          : CommandResult{ false, "nothing to redo" };
        };
        c.enabled = [](const CommandContext& ctx) {
            return ctx.scene_undo != nullptr && ctx.scene_undo->can_redo();
        };
        reg.add(cardinal::move(c));
    }
    // camera.focus_selection — aim the scene camera at the centroid of the
    // selected scene entities (collapses the host's 3 inline focus copies).
    {
        Command c;
        c.id = "camera.focus_selection"; c.label = "Focus Selection";
        c.run = [](CommandContext& ctx) -> CommandResult {
            if (ctx.scene == nullptr)            return { false, "no scene" };
            if (ctx.scene_selection.empty())     return { false, "empty selection" };
            cardinal::core::Vec3 sum{ 0, 0, 0 };
            cardinal::u32 n = 0;
            for (const auto id : ctx.scene_selection) {
                if (auto* e = ctx.scene->find_by_id(id)) {
                    sum.x += e->transform.translation.x;
                    sum.y += e->transform.translation.y;
                    sum.z += e->transform.translation.z;
                    ++n;
                }
            }
            if (n == 0) return { false, "no entities found" };
            const float inv = 1.0f / static_cast<float>(n);
            ctx.scene->camera().target = { sum.x * inv, sum.y * inv, sum.z * inv };
            ctx.result_count = n;
            return { true, {} };
        };
        c.enabled = [](const CommandContext& ctx) {
            return ctx.scene != nullptr && !ctx.scene_selection.empty();
        };
        reg.add(cardinal::move(c));
    }
}

}  // namespace cardinal::cmd
