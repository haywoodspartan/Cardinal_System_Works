// =============================================================================
// Cardinal UI — Widget base implementation.
// =============================================================================
#include <cardinal/cui/widget.hpp>

namespace cardinal::cui {

Widget* Widget::hit_test(Vec2 pt) noexcept {
    if (!visible_ || !rect_.contains(pt)) return nullptr;

    // Children are painted front-to-back in insertion order, so the topmost
    // (last) child wins a hit — search back-to-front.
    for (auto it = children_.rbegin(); it != children_.rend(); ++it) {
        if (!*it) continue;
        Widget* hit = (*it)->hit_test(pt);
        if (hit != nullptr) return hit;
    }
    // A widget only claims the hit if it's interactive, enabled, and not set to
    // ignore input (PAUIBase's IGNORE / DISABLE semantics).
    return (interactive() && enabled_ && !ignore_input_) ? this : nullptr;
}

Widget* Widget::find(const cardinal::string& id) noexcept {
    if (!id_.empty() && id_ == id) return this;
    for (auto& c : children_) {
        if (!c) continue;
        if (Widget* w = c->find(id)) return w;
    }
    return nullptr;
}

void Widget::paint_tree(PaintContext& ctx) {
    if (!visible_) return;
    const float saved = ctx.alpha;
    ctx.alpha = saved * alpha_;     // cascade this widget's alpha to its subtree
    paint(ctx);                     // draws (via ctx.apply) + recurses children
    ctx.alpha = saved;
}

}  // namespace cardinal::cui
