#pragma once

// =============================================================================
// Cardinal UI — the Ui context.
//
// Owns the root widget + theme, drives the per-frame cycle, and routes input.
// Typical frame:
//     ui.layout({w, h});            // measure + arrange the tree
//     ui.update_input(input);       // hover / press / click / drag
//     ui.paint(draw_list);          // emit draw commands
// The draw list is then handed to a backend (RHI/ImGui/test). Hover + active
// tracking is retained across frames so controls light up correctly.
// =============================================================================

#include <cardinal/core/types.hpp>

#include "widget.hpp"
#include "theme.hpp"

namespace cardinal::cui {

class Ui {
public:
    Ui() : theme_(default_theme()) {}

    void         set_theme(const Theme& t) { theme_ = t; }
    const Theme& theme() const noexcept    { return theme_; }

    // Install the root widget (Ui takes ownership).
    Widget* set_root(cardinal::unique_ptr<Widget> root) {
        root_ = cardinal::move(root);
        return root_.get();
    }
    Widget* root() const noexcept { return root_.get(); }

    // Per-frame.
    void layout(Vec2 screen_size);
    void update_input(const InputState& in);
    void paint(DrawList& dl);

    // Interaction state (read by widgets at paint time).
    bool is_hovered(const Widget* w) const noexcept { return w != nullptr && w == hovered_; }
    bool is_active (const Widget* w) const noexcept { return w != nullptr && w == active_; }
    const Widget* hovered() const noexcept { return hovered_; }
    const Widget* active()  const noexcept { return active_; }

private:
    Theme                        theme_;
    cardinal::unique_ptr<Widget> root_;
    Widget*                      hovered_{nullptr};
    Widget*                      active_{nullptr};
    InputState                   prev_{};
    bool                         prev_valid_{false};
};

}  // namespace cardinal::cui
