#pragma once

// =============================================================================
// Cardinal — scene validation.
//
// validate_world(world) scans the authored actor world for common authoring
// mistakes — the "problems panel" every editor has, so a designer catches a
// missing mesh asset or a stray second active camera BEFORE hitting Play.
//
// Rules are deterministic + side-effect-free (pure read over the World), so
// they're unit-testable and safe to run every frame the panel is open. Each
// issue carries a severity, the offending actor (id + name, id 0 for
// scene-level issues), and a human message.
// =============================================================================

#include <cardinal/core/types.hpp>        // cardinal::string
#include <cardinal/core/containers.hpp>   // cardinal::vector

namespace cardinal::actor {

class World;

enum class Severity : u32 { Info = 0, Warning = 1, Error = 2 };
const char* severity_name(Severity s) noexcept;

struct ValidationIssue {
    Severity         severity{Severity::Info};
    u32              actor{0};         // actor id; 0 = scene-level
    cardinal::string actor_name;
    cardinal::string message;
};

// Scan the world. Per-actor issues come first (in actor order), then
// scene-level issues (duplicate names, multiple active cameras). The list is
// deterministic for a given world.
cardinal::vector<ValidationIssue> validate_world(const World& world);

// Count issues at or above a severity (e.g. count_issues(issues, Warning)).
u32 count_issues(const cardinal::vector<ValidationIssue>& issues, Severity min_severity);

// Apply the mechanically-safe fixes — the ones with an unambiguous correct
// answer, so they don't need designer intent:
//   * a zero scale axis  -> reset that axis to 1
//   * a non-finite / far-out-of-bounds position -> reset to the origin
//   * a duplicate actor name -> rename the later one "<name> (2)" etc.
// Issues that DO need a human decision (missing mesh asset, dead light,
// extra active camera, empty actor) are left alone. Bumps the revision when
// it changes anything (so it's one undo step). Returns the number of fixes
// applied.
u32 auto_fix_world(World& world);

}  // namespace cardinal::actor
