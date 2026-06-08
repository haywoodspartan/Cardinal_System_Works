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
// hovered/active state, so controls can pick their colors).
struct PaintContext {
    DrawList*    dl{nullptr};
    const Theme* theme{nullptr};
    const Ui*    ui{nullptr};
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

    const Rect& rect() const noexcept { return rect_; }
    bool visible() const noexcept { return visible_; }
    void set_visible(bool v) noexcept { visible_ = v; }

    // Deepest interactive widget containing pt (topmost child wins).
    Widget* hit_test(Vec2 pt) noexcept;

protected:
    void paint_children(PaintContext& ctx) {
        for (auto& c : children_) if (c && c->visible_) c->paint(ctx);
    }

    Rect rect_{};
    Vec2 measured_{};
    bool visible_{true};
    cardinal::vector<cardinal::unique_ptr<Widget>> children_;
};

}  // namespace cardinal::cui
