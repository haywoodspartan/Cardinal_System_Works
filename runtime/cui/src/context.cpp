// =============================================================================
// Cardinal UI — Ui context implementation.
// =============================================================================
#include <cardinal/cui/context.hpp>

namespace cardinal::cui {

void Ui::layout(Vec2 screen_size) {
    if (!root_) return;
    root_->measure(Constraints::tight(screen_size.x, screen_size.y));
    root_->arrange(Rect{ {0.0f, 0.0f}, screen_size });
}

void Ui::update_input(const InputState& in) {
    hovered_ = root_ ? root_->hit_test(in.mouse) : nullptr;

    // Route the mouse wheel to the deepest scroll container under the cursor.
    if (in.scroll != 0.0f && root_ != nullptr) {
        if (Widget* s = root_->hit_scroll(in.mouse)) s->on_scroll(in.scroll);
    }

    if (prev_valid_) {
        const bool pressed  =  in.mouse_down && !prev_.mouse_down;
        const bool released = !in.mouse_down &&  prev_.mouse_down;

        if (pressed) active_ = hovered_;

        // Continuous drag for the held widget (sliders track the cursor).
        if (in.mouse_down && active_ != nullptr) active_->on_drag(in.mouse);

        if (released) {
            // A click only fires if release lands on the same widget pressed.
            if (active_ != nullptr && active_ == hovered_) active_->on_click();
            active_ = nullptr;
        }
    }

    prev_       = in;
    prev_valid_ = true;
}

void Ui::paint(DrawList& dl) {
    if (!root_) return;
    PaintContext ctx;
    ctx.dl    = &dl;
    ctx.theme = &theme_;
    ctx.ui    = this;
    dl.rect_filled(root_->rect(), theme_.window_bg, 0.0f);   // backdrop (full alpha)
    root_->paint_tree(ctx);                                  // cascades alpha down
}

}  // namespace cardinal::cui
