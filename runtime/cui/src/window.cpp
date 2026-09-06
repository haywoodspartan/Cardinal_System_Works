// =============================================================================
// Cardinal UI — Window + WindowManager. See window.hpp.
// =============================================================================
#include <cardinal/cui/window.hpp>

namespace cardinal::cui {

namespace {
inline float maxf(float a, float b) noexcept { return a > b ? a : b; }
inline float minf(float a, float b) noexcept { return a < b ? a : b; }
inline float absf(float a) noexcept { return a < 0.0f ? -a : a; }
}  // namespace

// -----------------------------------------------------------------------------
// Window
// -----------------------------------------------------------------------------
Window::Window(cardinal::string title, Rect frame)
    : title_(cardinal::move(title)), frame_(frame) {}

Widget* Window::set_content(cardinal::unique_ptr<Widget> content) {
    children_.clear();
    content_ = nullptr;
    if (!content) return nullptr;
    content_ = add(cardinal::move(content));
    apply_content_visibility_();
    return content_;
}

void Window::set_collapsed(bool c) {
    if (!collapsible_ && c) return;
    collapsed_ = c;
    apply_content_visibility_();
}

void Window::apply_content_visibility_() {
    if (content_) content_->set_visible(!collapsed_);
}

void Window::close() {
    set_visible(false);
    drag_valid_ = false;
    if (manager_ && manager_->active_ == this) manager_->active_ = nullptr;
    if (on_close_) on_close_();
}

Rect Window::title_rect() const noexcept {
    return { rect_.pos, { rect_.size.x, minf(title_h_, rect_.size.y) } };
}

Rect Window::content_rect() const noexcept {
    if (collapsed_) return { { rect_.left(), rect_.top() + title_h_ }, { rect_.size.x, 0.0f } };
    return rect_.inset(border_, title_h_, border_, border_);
}

Rect Window::close_rect() const noexcept {
    const float s = title_h_ - 8.0f;
    return { { rect_.right() - 4.0f - s, rect_.top() + 4.0f }, { s, s } };
}

Rect Window::collapse_rect() const noexcept {
    const float s = title_h_ - 8.0f;
    const float x = closable_ ? (rect_.right() - 4.0f - s - 4.0f - s) : (rect_.right() - 4.0f - s);
    return { { x, rect_.top() + 4.0f }, { s, s } };
}

Rect Window::grip_rect() const noexcept {
    return { { rect_.right() - grip_, rect_.bottom() - grip_ }, { grip_, grip_ } };
}

bool Window::contains_widget(const Widget* w) const noexcept {
    if (w == nullptr) return false;
    if (w == this) return true;
    // Iterative subtree walk (no recursion on the public tree API).
    cardinal::vector<const Widget*> stack;
    for (const auto& c : children_) if (c) stack.push_back(c.get());
    while (!stack.empty()) {
        const Widget* cur = stack.back();
        stack.pop_back();
        if (cur == w) return true;
        for (const auto& c : cur->children()) if (c) stack.push_back(c.get());
    }
    return false;
}

Vec2 Window::measure(const Constraints& c) {
    Vec2 s = frame_.size;
    if (collapsed_) s.y = title_h_;
    s.x = c.clamp_w(maxf(s.x, min_size_.x));
    s.y = c.clamp_h(collapsed_ ? title_h_ : maxf(s.y, min_size_.y));
    measured_ = s;
    return s;
}

void Window::arrange(const Rect& r) {
    rect_ = r;
    if (collapsed_ || content_ == nullptr) return;
    const Rect cr = content_rect();
    content_->measure(Constraints::tight(cr.size.x, cr.size.y));
    content_->arrange(cr);
}

Window::Drag Window::classify_(Vec2 p) const noexcept {
    if (closable_    && close_rect().contains(p))    return Drag::Press;
    if (collapsible_ && collapse_rect().contains(p)) return Drag::Press;
    if (resizable_ && !docked() && !collapsed_) {
        if (grip_rect().contains(p)) return Drag::ResizeCorner;
        if (p.x >= rect_.right()  - edge_ && p.y > rect_.top() + title_h_) return Drag::ResizeRight;
        if (p.y >= rect_.bottom() - edge_) return Drag::ResizeBottom;
    }
    if (title_rect().contains(p)) return (movable_ && !docked()) ? Drag::Move : Drag::Press;
    return Drag::Press;
}

void Window::on_drag(Vec2 mouse) {
    if (!drag_valid_) {
        drag_valid_  = true;
        press_       = mouse;
        start_frame_ = frame_;
        drag_        = classify_(mouse);
        if (manager_) manager_->on_window_pressed_(this);
        if (on_activate_) on_activate_();
        return;
    }
    const Vec2 delta{ mouse.x - press_.x, mouse.y - press_.y };
    switch (drag_) {
    case Drag::Move: {
        Rect f = start_frame_;
        f.pos = { start_frame_.pos.x + delta.x, start_frame_.pos.y + delta.y };
        if (manager_) {
            const Rect m = manager_->rect();
            const float snap = manager_->snap_distance();
            if (absf(f.left()   - m.left())   <= snap) f.pos.x = m.left();
            if (absf(f.right()  - m.right())  <= snap) f.pos.x = m.right() - f.size.x;
            if (absf(f.top()    - m.top())    <= snap) f.pos.y = m.top();
            if (absf(f.bottom() - m.bottom()) <= snap) f.pos.y = m.bottom() - f.size.y;
            f = manager_->clamp_floating_(f);
        }
        frame_ = f;
        break;
    }
    case Drag::ResizeRight:
        frame_.size.x = maxf(min_size_.x, start_frame_.size.x + delta.x);
        break;
    case Drag::ResizeBottom:
        frame_.size.y = maxf(min_size_.y, start_frame_.size.y + delta.y);
        break;
    case Drag::ResizeCorner:
        frame_.size.x = maxf(min_size_.x, start_frame_.size.x + delta.x);
        frame_.size.y = maxf(min_size_.y, start_frame_.size.y + delta.y);
        break;
    default:
        break;
    }
}

void Window::on_click() {
    const bool was_press = drag_valid_ && drag_ == Drag::Press;
    const Vec2 p = press_;
    drag_valid_ = false;
    drag_       = Drag::None;
    if (!was_press) return;
    if (closable_ && close_rect().contains(p)) { close(); return; }
    if (collapsible_ && collapse_rect().contains(p)) set_collapsed(!collapsed_);
}

void Window::paint(PaintContext& ctx) {
    // The Ui only calls on_drag while we are the active widget: once it lets
    // go, the next press must start a fresh drag classification.
    if (ctx.ui != nullptr && !ctx.ui->is_active(this)) drag_valid_ = false;

    const bool is_active = (manager_ != nullptr) ? (manager_->active_ == this)
                                                 : (ctx.ui != nullptr && ctx.ui->is_active(this));
    const Theme& th = *ctx.theme;
    ctx.dl->rect_filled(rect_, ctx.apply(th.panel_bg), th.rounding);
    // Title bar.
    const Rect tr = title_rect();
    ctx.dl->rect_filled(tr, ctx.apply(is_active ? th.control_active : th.control), th.rounding);
    const float ty = tr.top() + (tr.height() - line_height(font_size_)) * 0.5f;
    ctx.dl->text({ tr.left() + 8.0f, ty }, title_, ctx.apply(th.text), font_size_);
    // Buttons: collapse ("-" / "+") and close ("x").
    if (collapsible_) {
        const Rect cr = collapse_rect();
        ctx.dl->rect_stroke(cr, ctx.apply(th.border), 1.0f, 2.0f);
        const float gx = cr.left() + (cr.width() - text_advance("-", font_size_)) * 0.5f;
        const float gy = cr.top()  + (cr.height() - line_height(font_size_)) * 0.5f;
        ctx.dl->text({ gx, gy }, collapsed_ ? "+" : "-", ctx.apply(th.text), font_size_);
    }
    if (closable_) {
        const Rect cr = close_rect();
        ctx.dl->rect_stroke(cr, ctx.apply(th.border), 1.0f, 2.0f);
        const float gx = cr.left() + (cr.width() - text_advance("x", font_size_)) * 0.5f;
        const float gy = cr.top()  + (cr.height() - line_height(font_size_)) * 0.5f;
        ctx.dl->text({ gx, gy }, "x", ctx.apply(th.text), font_size_);
    }
    // Content, clipped to the content area.
    if (!collapsed_) {
        const Rect cr = content_rect();
        ctx.dl->push_clip(cr);
        paint_children(ctx);
        ctx.dl->pop_clip();
        // Resize grip: three diagonal ticks in the corner.
        if (resizable_ && !docked()) {
            const Rect g = grip_rect();
            for (int i = 1; i <= 3; ++i) {
                const float o = static_cast<float>(i) * 3.5f;
                ctx.dl->line({ g.right() - o, g.bottom() - 1.0f }, { g.right() - 1.0f, g.bottom() - o },
                             ctx.apply(th.text_dim), 1.0f);
            }
        }
    }
    ctx.dl->rect_stroke(rect_, ctx.apply(is_active ? th.accent : th.border), 1.0f, th.rounding);
}

// -----------------------------------------------------------------------------
// WindowManager
// -----------------------------------------------------------------------------
Window* WindowManager::add_window(cardinal::unique_ptr<Window> w) {
    if (!w) return nullptr;
    w->manager_ = this;
    Window* raw = w.get();
    add(cardinal::move(w));
    active_ = raw;
    return raw;
}

cardinal::unique_ptr<Window> WindowManager::remove_window(Window* w) {
    for (auto it = children_.begin(); it != children_.end(); ++it) {
        if (it->get() != w) continue;
        cardinal::unique_ptr<Widget> owned = cardinal::move(*it);
        children_.erase(it);
        if (active_ == w) active_ = nullptr;
        w->manager_ = nullptr;
        return cardinal::unique_ptr<Window>(static_cast<Window*>(owned.release()));
    }
    return nullptr;
}

cardinal::vector<Window*> WindowManager::windows() const {
    cardinal::vector<Window*> out;
    out.reserve(children_.size());
    for (const auto& c : children_) if (c) out.push_back(static_cast<Window*>(c.get()));
    return out;
}

Window* WindowManager::front() const noexcept {
    for (auto it = children_.rbegin(); it != children_.rend(); ++it)
        if (*it && (*it)->visible()) return static_cast<Window*>(it->get());
    return nullptr;
}

void WindowManager::bring_to_front(Window* w) {
    if (w == nullptr) return;
    for (usize i = 0; i < children_.size(); ++i) {
        if (children_[i].get() != w) continue;
        cardinal::unique_ptr<Widget> owned = cardinal::move(children_[i]);
        children_.erase(children_.begin() + static_cast<long long>(i));
        children_.push_back(cardinal::move(owned));
        break;
    }
    active_ = w;
}

void WindowManager::on_window_pressed_(Window* w) { bring_to_front(w); }

void WindowManager::dock(Window* w, DockSide side) {
    if (w == nullptr || w->manager_ != this) return;
    if (!w->docked()) w->floating_frame_ = w->frame_;
    w->dock_side_  = side;
    w->dock_order_ = ++dock_seq_;
    if (side == DockSide::None) w->frame_ = w->floating_frame_;
}

void WindowManager::undock(Window* w) { dock(w, DockSide::None); }

void WindowManager::set_dock_thickness(DockSide side, float px) {
    const usize i = static_cast<usize>(side);
    if (i == 0 || i > 4) return;
    thickness_[i] = maxf(40.0f, px);
}

float WindowManager::dock_thickness(DockSide side) const noexcept {
    const usize i = static_cast<usize>(side);
    return (i == 0 || i > 4) ? 0.0f : thickness_[i];
}

Rect WindowManager::dock_rect(DockSide side) const noexcept {
    const usize i = static_cast<usize>(side);
    return (i == 0 || i > 4) ? Rect{} : strips_[i];
}

Vec2 WindowManager::measure(const Constraints& c) {
    // Fills the space it is offered (like Canvas); a loose max falls back to the min.
    const Vec2 s{ c.max_w < 1.0e8f ? c.max_w : c.min_w,
                  c.max_h < 1.0e8f ? c.max_h : c.min_h };
    measured_ = s;
    return s;
}

Rect WindowManager::clamp_floating_(Rect f) const noexcept {
    const float keep = 40.0f;   // pixels of the title bar that must stay reachable
    const float min_x = rect_.left() - f.size.x + keep;
    const float max_x = rect_.right() - keep;
    const float min_y = rect_.top();
    const float max_y = maxf(rect_.top(), rect_.bottom() - 24.0f);
    if (f.pos.x < min_x) f.pos.x = min_x;
    if (f.pos.x > max_x) f.pos.x = max_x;
    if (f.pos.y < min_y) f.pos.y = min_y;
    if (f.pos.y > max_y) f.pos.y = max_y;
    return f;
}

void WindowManager::arrange(const Rect& r) {
    rect_ = r;
    // Which strips are occupied?
    bool used[5] = { false, false, false, false, false };
    for (const auto& c : children_) {
        if (!c || !c->visible()) continue;
        const auto* w = static_cast<const Window*>(c.get());
        if (w->docked()) used[static_cast<usize>(w->dock_side_)] = true;
    }
    const float tt = used[3] ? thickness_[3] : 0.0f;   // Top
    const float tb = used[4] ? thickness_[4] : 0.0f;   // Bottom
    const float tl = used[1] ? thickness_[1] : 0.0f;   // Left
    const float trr = used[2] ? thickness_[2] : 0.0f;  // Right
    const float mid_h = maxf(0.0f, r.height() - tt - tb);
    strips_[3] = { { r.left(), r.top() }, { r.width(), tt } };
    strips_[4] = { { r.left(), r.bottom() - tb }, { r.width(), tb } };
    strips_[1] = { { r.left(), r.top() + tt }, { tl, mid_h } };
    strips_[2] = { { r.right() - trr, r.top() + tt }, { trr, mid_h } };
    client_    = { { r.left() + tl, r.top() + tt }, { maxf(0.0f, r.width() - tl - trr), mid_h } };

    // Docked windows split their strip evenly along its long axis.
    for (usize side = 1; side <= 4; ++side) {
        cardinal::vector<Window*> in_strip;
        for (const auto& c : children_) {
            if (!c || !c->visible()) continue;
            auto* w = static_cast<Window*>(c.get());
            if (static_cast<usize>(w->dock_side_) == side) in_strip.push_back(w);
        }
        if (in_strip.empty()) continue;
        // Slot order = docking order, NOT z-order (raising a docked window
        // must not shuffle the strip). Insertion sort: strips are tiny.
        for (usize i = 1; i < in_strip.size(); ++i) {
            Window* w = in_strip[i];
            usize j = i;
            while (j > 0 && in_strip[j - 1]->dock_order_ > w->dock_order_) { in_strip[j] = in_strip[j - 1]; --j; }
            in_strip[j] = w;
        }
        const Rect strip = strips_[side];
        const bool vertical_split = (side == 1 || side == 2);   // Left / Right stack vertically
        const float n = static_cast<float>(in_strip.size());
        for (usize i = 0; i < in_strip.size(); ++i) {
            Rect slot = strip;
            if (vertical_split) {
                slot.size.y = strip.height() / n;
                slot.pos.y  = strip.top() + slot.size.y * static_cast<float>(i);
            } else {
                slot.size.x = strip.width() / n;
                slot.pos.x  = strip.left() + slot.size.x * static_cast<float>(i);
            }
            in_strip[i]->arrange(slot);
        }
    }
    // Floating windows at their (clamped) frames.
    for (const auto& c : children_) {
        if (!c || !c->visible()) continue;
        auto* w = static_cast<Window*>(c.get());
        if (w->docked()) continue;
        Rect f = clamp_floating_(w->frame_);
        f.size.x = maxf(f.size.x, w->min_size_.x);
        f.size.y = w->collapsed_ ? w->title_h_ : maxf(f.size.y, w->min_size_.y);
        w->frame_.pos = f.pos;                 // keep the stored frame on-screen too
        w->arrange(f);
    }
}

void WindowManager::sync_z_order_(const Ui& ui) {
    const Widget* a = ui.active();
    if (a != nullptr && a != last_ui_active_) {
        // Topmost first: the frontmost window containing the pressed widget wins.
        for (auto it = children_.rbegin(); it != children_.rend(); ++it) {
            auto* w = static_cast<Window*>(it->get());
            if (w && w->visible() && w->contains_widget(a)) { bring_to_front(w); break; }
        }
    }
    last_ui_active_ = a;
}

void WindowManager::paint(PaintContext& ctx) {
    if (ctx.ui != nullptr) sync_z_order_(*ctx.ui);
    paint_children(ctx);
}

}  // namespace cardinal::cui
