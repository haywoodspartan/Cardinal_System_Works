// =============================================================================
// Cardinal — Undo stack implementation.
// =============================================================================
#include <cardinal/edit/undo.hpp>
#include <cardinal/core/utility.hpp>   // cardinal::move

namespace cardinal::edit {

void UndoStack::truncate_redo() {
    if (cursor_ < history_.size()) history_.resize(cursor_);
}
void UndoStack::enforce_capacity() {
    if (history_.size() <= capacity_) return;
    const usize drop = history_.size() - capacity_;
    history_.erase(history_.begin(), history_.begin() + drop);
    cursor_ = (cursor_ > drop) ? cursor_ - drop : 0;
}

void UndoStack::push_executed(Command cmd) {
    truncate_redo();
    history_.push_back(cardinal::move(cmd));
    cursor_ = history_.size();
    enforce_capacity();
}

void UndoStack::push_and_apply(Command cmd) {
    if (cmd.apply) cmd.apply();
    push_executed(cardinal::move(cmd));
}

bool UndoStack::undo() {
    if (!can_undo()) return false;
    --cursor_;
    auto& c = history_[cursor_];
    if (c.revert) c.revert();
    return true;
}

bool UndoStack::redo() {
    if (!can_redo()) return false;
    auto& c = history_[cursor_];
    if (c.apply) c.apply();
    ++cursor_;
    return true;
}

const char* UndoStack::next_undo_label() const noexcept {
    if (!can_undo()) return "";
    return history_[cursor_ - 1].label.c_str();
}
const char* UndoStack::next_redo_label() const noexcept {
    if (!can_redo()) return "";
    return history_[cursor_].label.c_str();
}

}  // namespace cardinal::edit
