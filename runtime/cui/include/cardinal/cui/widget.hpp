#pragma once

// =============================================================================
// Cardinal UI — retained widget base.
//
// Layout is two-pass (Flutter/WPF style):
//   1. measure(constraints) -> desired size
//   2. arrange(final_rect)  -> commit bounds + position children
// then paint(ctx) emits draw commands. Input is routed by the Ui via
// interactive()/on_click()/on_drag(); a widget never reads raw input itself.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/utility.hpp>

#include "geometry.hpp"
#include "draw.hpp"
#include "theme.hpp"

namespace cardinal::cui {

class Ui;   // forward — paint context carries a const Ui* for interaction state

// Per-frame input snapshot fed to Ui::update_input().
struct InputState {
    Vec2  mouse{};
    bool  mouse_down{false};
    float scroll{0.0f};
};

// Everything paint() needs: the draw list, the theme, and the Ui (for
// hovered/active state, so controls can pick their colors). `alpha` is the
// cumulative ancestor alpha — paints multiply their colors through apply() so a
// faded parent fades its whole subtree (PAUIBase's SetAlphaChild cascade, done
// without per-widget plumbing).
struct PaintContext {
    DrawList*    dl{nullptr};
    const Theme* theme{nullptr};
    const Ui*    ui{nullptr};
    float        alpha{1.0f};

    Color apply(Color c) const noexcept {
        float a = static_cast<float>(c.a) * alpha;
        c.a = static_cast<u8>(a < 0.0f ? 0.0f : (a > 255.0f ? 255.0f : a));
        return c;
    }
};

class Widget {
public:
    virtual ~Widget() = default;

    // --- Layout -------------------------------------------------------
    virtual Vec2 measure(const Constraints& c) = 0;
    virtual void arrange(const Rect& final_rect) { rect_ = final_rect; }

    // --- Paint --------------------------------------------------------
    virtual void paint(PaintContext& ctx) = 0;

    // --- Interaction (default: inert container) -----------------------
    virtual bool interactive() const noexcept { return false; }
    virtual void on_click() {}
    virtual void on_drag(Vec2 /*mouse*/) {}

    // --- Tree ---------------------------------------------------------
    Widget* add(cardinal::unique_ptr<Widget> child) {
        Widget* raw = child.get();
        children_.push_back(cardinal::move(child));
        return raw;
    }
    const cardinal::vector<cardinal::unique_ptr<Widget>>& children() const noexcept {
        return children_;
    }

    // --- Identity (PAUIBase id / FindControl, our own clean form) ------
    void                    set_id(cardinal::string id) { id_ = cardinal::move(id); }
    const cardinal::string& id() const noexcept { return id_; }
    // First widget in this subtree (self included) whose id matches; else null.
    Widget*                 find(const cardinal::string& id) noexcept;

    // --- State (PAUIBase Show / Enable / Ignore) ----------------------
    const Rect& rect() const noexcept { return rect_; }
    bool visible() const noexcept { return visible_; }
    void set_visible(bool v) noexcept { visible_ = v; }
    bool enabled() const noexcept { return enabled_; }
    void set_enabled(bool e) noexcept { enabled_ = e; }
    bool ignore_input() const noexcept { return ignore_input_; }
    void set_ignore_input(bool i) noexcept { ignore_input_ = i; }
    float alpha() const noexcept { return alpha_; }
    void  set_alpha(float a) noexcept { alpha_ = a; }

    // Deepest interactive, enabled, non-ignored widget containing pt.
    Widget* hit_test(Vec2 pt) noexcept;

    // Paint this widget + subtree, cascading alpha down the context. Hosts call
    // this (not paint()) on the root; paint_children calls it on each child.
    void paint_tree(PaintContext& ctx);

protected:
    void paint_children(PaintContext& ctx) {
        for (auto& c : children_) if (c) c->paint_tree(ctx);
    }

    Rect             rect_{};
    Vec2             measured_{};
    bool             visible_{true};
    bool             enabled_{true};
    bool             ignore_input_{false};
    float            alpha_{1.0f};
    cardinal::string id_;
    cardinal::vector<cardinal::unique_ptr<Widget>> children_;
};

}  // namespace cardinal::cui
