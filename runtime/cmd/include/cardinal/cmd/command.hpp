#pragma once

// =============================================================================
// Cardinal — unified editor command bus (cardinal::cmd).
//
// One registry where engine operations are registered as named, callable
// Commands. Every UI entry point — menu bar, toolbar, context menus, hotkeys,
// and (eventually) a command palette + console — dispatches by string id
// through this ONE table, instead of each panel/host hand-wiring its own call
// into the engine. Benefits: a single dedup'd dispatch surface, headless
// testability (invoke a command without the GUI), and — critically — actions
// that take a CAPTURED engine state instead of re-deriving it ad hoc.
//
// This module deliberately has NO ui / imgui / swapchain dependency, so a test
// can construct a CommandContext over real engine fixtures + a synthetic
// view/proj and dispatch() a command, asserting on the result. Per the
// foundation rule, std headers stay in cardinal::core; this header uses the
// cardinal:: aliases exclusively.
// =============================================================================

#include <cardinal/core/types.hpp>        // cardinal::string / function / usize
#include <cardinal/core/std/containers.hpp>   // cardinal::vector
#include <cardinal/core/dense_map.hpp>    // cardinal::dense_map (registry storage)
#include <cardinal/core/string/string_id.hpp> // cardinal::StringId (registry key)
#include <cardinal/core/math/math.hpp>         // cardinal::core::Mat4 / Vec3

namespace cardinal::actor { class World; }
namespace cardinal::scene { class Scene; }
namespace cardinal::level { class AssetPlacement; }
namespace cardinal::edit  { class UndoStack; }
namespace cardinal::rhi   { class Device; }

namespace cardinal::cmd {

// A render view-projection captured at the EXACT site/frame the scene was
// rasterised for a viewport. A viewport pick unprojects with THESE matrices
// (not ones re-derived from the live camera/aspect at click time), so the
// pick ray is bit-identical to what was drawn — eliminating the aspect /
// camera-staleness divergence that skews placement off the cursor.
struct ViewProj {
    cardinal::core::Mat4 view{};
    cardinal::core::Mat4 proj{};
    bool                 valid{false};
};

// Everything a command may act on. A TRANSIENT bundle of references valid only
// for the duration of a single dispatch() call — never cache it (the refs and
// the captured viewport dangle after the frame).
struct CommandContext {
    cardinal::actor::World*           world{nullptr};
    cardinal::scene::Scene*           scene{nullptr};
    cardinal::level::AssetPlacement*  placement{nullptr};
    cardinal::edit::UndoStack*        scene_undo{nullptr};   // edit.undo / edit.redo
    cardinal::rhi::Device*            device{nullptr};

    // Viewport pick inputs (click-driven commands).
    ViewProj viewport{};   // captured render matrices for the hovered viewport
    float    ndc_x{0.0f};  // panel-local NDC of the click (-1..+1)
    float    ndc_y{0.0f};
    // Direct world position for place_asset's non-click mode (menu spawn).
    // Used when `viewport` is invalid (no click to unproject).
    cardinal::core::Vec3 place_position{0.0f, 0.0f, 0.0f};

    // Typed inputs (no bag-of-any).
    cardinal::string active_asset_id;   // world.place_asset
    bool             snap_enabled{false};
    float            snap_step{0.0f};

    // Selection + layout inputs (actor.* / layout.* commands act on these).
    cardinal::vector<cardinal::u32> selection;       // selected ACTOR ids
    cardinal::vector<cardinal::u32> scene_selection; // selected SCENE entity ids (camera.* / scene.*)
    int   axis{0};         // 0=X 1=Y 2=Z   (layout.align / layout.distribute)
    int   align_mode{1};   // 0=Min 1=Center 2=Max (layout.align)
    float grid_step{1.0f}; // layout.snap

    // Outputs a command may fill.
    cardinal::u32        result_actor{0};
    cardinal::u32        result_entity{0};
    cardinal::u32        result_count{0};   // # affected (bulk/layout ops)
    cardinal::core::Vec3 result_hit{0.0f, 0.0f, 0.0f};
};

struct CommandResult {
    bool             ok{false};
    cardinal::string message;   // diagnostic on failure (or empty)
};

struct Command {
    cardinal::string id;      // dotted, e.g. "world.place_asset"
    cardinal::string label;   // human label, e.g. "Place Asset"
    cardinal::function<CommandResult(CommandContext&)> run;
    // Optional predicate: can this command run in the given context? Used to
    // grey out unavailable entries in the palette / menus. Null == always
    // enabled. dispatch() does NOT consult it — a disabled command still
    // fails gracefully via its own run() guards; this is purely for UI.
    cardinal::function<bool(const CommandContext&)> enabled;
};

class CommandRegistry {
public:
    void  add(Command cmd);                                    // last write wins
    bool  has(const cardinal::string& id) const;
    const Command* find(const cardinal::string& id) const;     // nullptr if absent
    // Dispatch by id. Unknown id / empty handler -> ok=false (never throws),
    // so callers can route hotkeys/console blindly.
    CommandResult dispatch(const cardinal::string& id, CommandContext& ctx) const;
    // Is the command runnable in this context? false for an unknown id; for a
    // known command with no `enabled` predicate -> true (always enabled).
    bool is_enabled(const cardinal::string& id, const CommandContext& ctx) const;
    cardinal::vector<cardinal::string> ids() const;            // sorted (palette/console)
    cardinal::usize size() const noexcept { return commands_.size(); }

private:
    // Keyed by the StringId hash of the (compile-time-stable) command id, not
    // the string itself — find/has/dispatch/is_enabled become a u64-keyed
    // O(1) DenseMap probe instead of string hashing + node chasing. The public
    // API stays string-based (callers pass "world.undo" etc.); the id string
    // is also kept inside each Command for ids() + display. 64-bit FNV over a
    // controlled, dotted id namespace makes collisions a non-issue.
    cardinal::dense_map<cardinal::StringId, Command> commands_;
};

// Pure, Device-free placement math (the offset bug lived here, fed the wrong
// matrices). Unproject an NDC point with the captured view/proj and intersect
// the world Y = plane_y plane, applying optional grid snap. Returns false if
// the matrices are invalid or the ray is parallel to the plane.
bool screen_to_ground(const ViewProj& vp, float ndc_x, float ndc_y,
                      float plane_y, bool snap, float snap_step,
                      cardinal::core::Vec3* out_hit);

// Install the built-in editor commands (world.place_asset, ...) into `reg`.
void register_builtin_commands(CommandRegistry& reg);

}  // namespace cardinal::cmd
