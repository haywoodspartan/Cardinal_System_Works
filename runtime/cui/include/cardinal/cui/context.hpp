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
    bool is_focused(const Widget* w) const noexcept { return w != nullptr && w == focused_; }
    const Widget* hovered() const noexcept { return hovered_; }
    const Widget* active()  const noexcept { return active_; }
    const Widget* focused() const noexcept { return focused_; }

    // Keyboard focus. Pressing a focusable widget focuses it; pressing anything
    // else (or Escape) clears focus. NOTE: like hovered_/active_, this is a
    // non-owning pointer — clear focus before destroying focused widgets.
    void set_focus(Widget* w) {
        if (w == focused_) return;
        if (focused_ != nullptr) focused_->on_focus_changed(false);
        focused_ = w;
        if (focused_ != nullptr) focused_->on_focus_changed(true);
    }
    void clear_focus() { set_focus(nullptr); }

    // Popup priority (host-managed). A widget with an open dropdown (Combo,
    // MenuBar) extends its rect_ over later siblings, but tree hit order gives
    // the LAST child the hit and paint order draws it UNDER later siblings.
    // Point the Ui at the popped widget and it wins hit-testing and repaints
    // on top of the tree (small overdraw of the widget itself — acceptable).
    // Hosts set this each frame, e.g.:
    //     ui.set_popup(combo->open() ? combo : nullptr);
    // Non-owning, like focused_ — clear before destroying the widget.
    void    set_popup(Widget* w) noexcept { popup_ = w; }
    Widget* popup() const noexcept        { return popup_; }

private:
    Theme                        theme_;
    cardinal::unique_ptr<Widget> root_;
    Widget*                      hovered_{nullptr};
    Widget*                      active_{nullptr};
    Widget*                      focused_{nullptr};
    Widget*                      popup_{nullptr};
    InputState                   prev_{};
    bool                         prev_valid_{false};
};

}  // namespace cardinal::cui
