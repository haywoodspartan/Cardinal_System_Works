#pragma once

// =============================================================================
// Cardinal — Undo / Redo command stack.
//
// `UndoStack` is a bounded LIFO of `Command`s. Each command captures the two
// inverse operations needed to apply / undo a single user-facing action:
//
//     edit::Command cmd;
//     cmd.label   = "Move entity";
//     cmd.apply   = [=]{ entity.transform.translation = new_pos; };
//     cmd.revert  = [=]{ entity.transform.translation = old_pos; };
//     stack.push_executed(cardinal::move(cmd));   // records that apply already ran
//
// `push_executed` is the common case for actions performed via direct UI
// (the user dragged the gizmo, the engine already moved the entity, we
// just need to remember how to undo it). `push_and_apply` is the variant
// used by command-pattern callers that haven't yet performed the action.
//
// Capacity is bounded; pushing past the cap drops the oldest record.
// Pushing while in the middle of an undo chain truncates the redo half.
// =============================================================================

#include <cardinal/core/types.hpp>       // cardinal::function, cardinal::string
#include <cardinal/core/containers.hpp>  // cardinal::vector

namespace cardinal::edit {

struct Command {
    cardinal::string             label;
    cardinal::function<void()>   apply;
    cardinal::function<void()>   revert;
};

class UndoStack {
public:
    explicit UndoStack(usize capacity = 256) noexcept : capacity_(capacity) {}

    // Push a command whose `apply` has ALREADY been executed by the host.
    // The history pointer advances; any pending redo is dropped.
    void push_executed(Command cmd);

    // Push a command and execute its `apply` immediately. Same history
    // bookkeeping as push_executed.
    void push_and_apply(Command cmd);

    // Returns true on success (i.e. there was something to undo / redo).
    bool undo();
    bool redo();

    bool can_undo() const noexcept { return cursor_ > 0; }
    bool can_redo() const noexcept { return cursor_ < history_.size(); }

    // For UI hover-text — returns the next operation's label (or empty).
    const char* next_undo_label() const noexcept;
    const char* next_redo_label() const noexcept;

    void clear() noexcept { history_.clear(); cursor_ = 0; }
    usize size() const noexcept { return history_.size(); }

private:
    void truncate_redo();
    void enforce_capacity();

    cardinal::vector<Command> history_;
    usize                cursor_{0};   // 0..history_.size(); commands [0, cursor) are applied
    usize                capacity_;
};

}  // namespace cardinal::edit
