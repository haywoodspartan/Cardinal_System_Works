// =============================================================================
// Cardinal — WorldHistory implementation.
// =============================================================================
#include <cardinal/serial/world_history.hpp>
#include <cardinal/serial/serial.hpp>

#include <cardinal/game/game.hpp>
#include <cardinal/actor/world.hpp>
#include <cardinal/core/std/utility.hpp>   // cardinal::move

namespace cardinal::serial {

WorldHistory::WorldHistory(usize max_depth) noexcept
    : max_depth_(max_depth < 2 ? 2 : max_depth) {}

bool WorldHistory::capture(const cardinal::actor::World& world) {
    cardinal::string snap = serialize_world(world);

    // No-op if the world is identical to the active checkpoint — avoids
    // piling up duplicate states (e.g. capture() called on an unchanged
    // scene).
    if (!snaps_.empty() && snap == snaps_[cursor_]) return false;

    // Drop the redo tail (everything after the cursor): capturing a new
    // state forks history, discarding the old forward branch.
    if (!snaps_.empty()) {
        while (snaps_.size() > cursor_ + 1) snaps_.pop_back();
    }

    snaps_.push_back(cardinal::move(snap));
    cursor_ = snaps_.size() - 1;

    // Enforce the depth bound — drop oldest snapshots, shifting the cursor.
    while (snaps_.size() > max_depth_) {
        snaps_.erase(snaps_.begin());
        if (cursor_ > 0) --cursor_;
    }
    return true;
}

bool WorldHistory::can_undo() const noexcept {
    return !snaps_.empty() && cursor_ > 0;
}
bool WorldHistory::can_redo() const noexcept {
    return !snaps_.empty() && cursor_ + 1 < snaps_.size();
}

bool WorldHistory::undo(cardinal::game::Game& game) {
    if (!can_undo()) return false;
    --cursor_;
    deserialize_world(game, snaps_[cursor_], /*replace_existing=*/true);
    return true;
}

bool WorldHistory::redo(cardinal::game::Game& game) {
    if (!can_redo()) return false;
    ++cursor_;
    deserialize_world(game, snaps_[cursor_], /*replace_existing=*/true);
    return true;
}

void WorldHistory::clear() noexcept {
    snaps_.clear();
    cursor_ = 0;
}

}  // namespace cardinal::serial
