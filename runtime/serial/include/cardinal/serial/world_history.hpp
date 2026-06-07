#pragma once

// =============================================================================
// Cardinal — WorldHistory: snapshot-based undo / redo for the actor world.
//
// Built on serialize_world / deserialize_world (in-memory text snapshots).
// The editor captures a checkpoint at meaningful boundaries (a baseline at
// load, before a risky bulk edit, before Play, after a Load); undo / redo
// step the live world back and forth through those checkpoints.
//
// Model — a linear stack of snapshots with a cursor at the ACTIVE state:
//
//   capture(world): snapshot the world. Drops any redo tail (states after
//     the cursor), appends the snapshot, moves the cursor to it. A snapshot
//     identical to the current one is a no-op (no duplicate checkpoints).
//     Oldest snapshots are dropped past max_depth.
//   undo(game): cursor-- and restore that snapshot. False at the bottom.
//   redo(game): cursor++ and restore that snapshot. False at the top.
//
// This is CHECKPOINT undo, not per-keystroke: it restores to the last
// captured state, so the host decides granularity by when it calls
// capture(). Snapshots are full-world text (cheap for editor-scale scenes).
// =============================================================================

#include <cardinal/core/types.hpp>        // cardinal::string
#include <cardinal/core/std/containers.hpp>   // cardinal::vector

namespace cardinal::actor { class World; }
namespace cardinal::game  { class Game;  }

namespace cardinal::serial {

class WorldHistory {
public:
    explicit WorldHistory(usize max_depth = 64) noexcept;

    // Capture the current world as a new checkpoint (see model above).
    // Returns true if a NEW checkpoint was added (false on the no-op dedup).
    bool capture(const cardinal::actor::World& world);

    // Step the live world to the previous / next checkpoint. Returns false
    // when already at the bottom / top of the stack.
    bool undo(cardinal::game::Game& game);
    bool redo(cardinal::game::Game& game);

    bool  can_undo() const noexcept;   // a previous checkpoint exists
    bool  can_redo() const noexcept;   // a forward checkpoint exists
    usize depth()   const noexcept { return snaps_.size(); }
    usize cursor()  const noexcept { return cursor_; }       // active index
    bool  empty()   const noexcept { return snaps_.empty(); }
    void  clear() noexcept;

private:
    cardinal::vector<cardinal::string> snaps_;
    usize cursor_{0};        // index of the active checkpoint (valid iff !empty)
    usize max_depth_{64};
};

}  // namespace cardinal::serial
