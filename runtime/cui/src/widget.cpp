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
    return interactive() ? this : nullptr;
}

}  // namespace cardinal::cui
