// =============================================================================
// Cardinal UI — core widget implementations.
// =============================================================================
#include <cardinal/cui/widgets.hpp>
#include <cardinal/cui/context.hpp>

namespace cardinal::cui {

namespace {
inline float maxf(float a, float b) noexcept { return a > b ? a : b; }
inline float clamp01(float t) noexcept { return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t); }
}  // namespace

// -----------------------------------------------------------------------------
// Stack
// -----------------------------------------------------------------------------
Vec2 Stack::measure(const Constraints& c) {
    float main = 0.0f, cross = 0.0f;
    int n = 0;
    for (auto& ch : children_) {
        if (!ch || !ch->visible()) continue;
        const Vec2 s = ch->measure(c);
        if (axis_ == Axis::Vertical) { main += s.y; cross = maxf(cross, s.x); }
        else                         { main += s.x; cross = maxf(cross, s.y); }
        ++n;
    }
    if (n > 1) main += spacing_ * static_cast<float>(n - 1);

    Vec2 sz = (axis_ == Axis::Vertical) ? Vec2{cross, main} : Vec2{main, cross};
    sz.x = c.clamp_w(sz.x);
    sz.y = c.clamp_h(sz.y);
    measured_ = sz;
    return sz;
}

void Stack::arrange(const Rect& r) {
    rect_ = r;
    float cursor = (axis_ == Axis::Vertical) ? r.top() : r.left();
    for (auto& ch : children_) {
        if (!ch || !ch->visible()) continue;
        const Vec2 s = ch->measure(Constraints::loose(r.width(), r.height()));
        Rect cr;
        if (axis_ == Axis::Vertical) {
            float w = (cross_ == Align::Stretch) ? r.width() : s.x;
            float x = r.left();
            if      (cross_ == Align::Center) x = r.left() + (r.width() - w) * 0.5f;
            else if (cross_ == Align::End)    x = r.right() - w;
            cr = { { x, cursor }, { w, s.y } };
            cursor += s.y + spacing_;
        } else {
            float h = (cross_ == Align::Stretch) ? r.height() : s.y;
            float y = r.top();
            if      (cross_ == Align::Center) y = r.top() + (r.height() - h) * 0.5f;
            else if (cross_ == Align::End)    y = r.bottom() - h;
            cr = { { cursor, y }, { s.x, h } };
            cursor += s.x + spacing_;
        }
        ch->arrange(cr);
    }
}

// -----------------------------------------------------------------------------
// Panel
// -----------------------------------------------------------------------------
Vec2 Panel::measure(const Constraints& c) {
    const Constraints inner = c.deflate(pad_);
    Vec2 child{0.0f, 0.0f};
    for (auto& ch : children_) {
        if (!ch || !ch->visible()) continue;
        const Vec2 s = ch->measure(inner);
        child.x = maxf(child.x, s.x);
        child.y = maxf(child.y, s.y);
    }
    Vec2 sz{ child.x + pad_.horiz(), child.y + pad_.vert() };
    sz.x = c.clamp_w(sz.x);
    sz.y = c.clamp_h(sz.y);
    measured_ = sz;
    return sz;
}

void Panel::arrange(const Rect& r) {
    rect_ = r;
    const Rect inner = r.inset(pad_.l, pad_.t, pad_.r, pad_.b);
    for (auto& ch : children_) if (ch && ch->visible()) ch->arrange(inner);
}

void Panel::paint(PaintContext& ctx) {
    ctx.dl->rect_filled(rect_, ctx.apply(bg_), ctx.theme->rounding);
    ctx.dl->rect_stroke(rect_, ctx.apply(ctx.theme->border), 1.0f, ctx.theme->rounding);
    paint_children(ctx);
}

// -----------------------------------------------------------------------------
// Label
// -----------------------------------------------------------------------------
Vec2 Label::measure(const Constraints& c) {
    Vec2 s{ text_advance(text_, font_size_), line_height(font_size_) };
    s.x = c.clamp_w(s.x);
    s.y = c.clamp_h(s.y);
    measured_ = s;
    return s;
}

void Label::paint(PaintContext& ctx) {
    ctx.dl->text(rect_.pos, text_, ctx.apply(ctx.theme->text), font_size_);
}

// -----------------------------------------------------------------------------
// Button
// -----------------------------------------------------------------------------
Vec2 Button::measure(const Constraints& c) {
    Vec2 s{ text_advance(text_, font_size_) + pad_.horiz(),
            line_height(font_size_) + pad_.vert() };
    s.x = c.clamp_w(s.x);
    s.y = c.clamp_h(s.y);
    measured_ = s;
    return s;
}

void Button::paint(PaintContext& ctx) {
    Color bg = ctx.theme->control;
    if      (ctx.ui->is_active(this))  bg = ctx.theme->control_active;
    else if (ctx.ui->is_hovered(this)) bg = ctx.theme->control_hot;
    ctx.dl->rect_filled(rect_, ctx.apply(bg), ctx.theme->rounding);
    const Vec2 tp{ rect_.pos.x + pad_.l, rect_.pos.y + pad_.t };
    ctx.dl->text(tp, text_, ctx.apply(ctx.theme->text), font_size_);
}

// -----------------------------------------------------------------------------
// Checkbox
// -----------------------------------------------------------------------------
Vec2 Checkbox::measure(const Constraints& c) {
    Vec2 s{ box_ + gap_ + text_advance(label_, font_size_),
            maxf(box_, line_height(font_size_)) };
    s.x = c.clamp_w(s.x);
    s.y = c.clamp_h(s.y);
    measured_ = s;
    return s;
}

void Checkbox::paint(PaintContext& ctx) {
    const Rect box{ rect_.pos, { box_, box_ } };
    const Color bg = ctx.ui->is_hovered(this) ? ctx.theme->control_hot : ctx.theme->control;
    ctx.dl->rect_filled(box, ctx.apply(bg), 2.0f);
    ctx.dl->rect_stroke(box, ctx.apply(ctx.theme->border), 1.0f, 2.0f);
    if (value_ != nullptr && *value_) {
        ctx.dl->rect_filled(box.inset(3, 3, 3, 3), ctx.apply(ctx.theme->accent), 1.0f);
    }
    const Vec2 tp{ rect_.pos.x + box_ + gap_,
                   rect_.pos.y + (box_ - line_height(font_size_)) * 0.5f };
    ctx.dl->text(tp, label_, ctx.apply(ctx.theme->text), font_size_);
}

// -----------------------------------------------------------------------------
// Slider
// -----------------------------------------------------------------------------
Vec2 Slider::measure(const Constraints& c) {
    // Take the offered width when finite, else a sensible default.
    const float w = (c.max_w < 1e8f) ? c.max_w : 160.0f;
    Vec2 s{ w, height_ };
    s.x = c.clamp_w(s.x);
    s.y = c.clamp_h(s.y);
    measured_ = s;
    return s;
}

void Slider::on_drag(Vec2 mouse) {
    if (value_ == nullptr || rect_.width() <= 0.0f) return;
    const float t = clamp01((mouse.x - rect_.left()) / rect_.width());
    *value_ = min_ + t * (max_ - min_);
}

void Slider::paint(PaintContext& ctx) {
    const Rect track{ { rect_.left(), rect_.center().y - 3.0f }, { rect_.width(), 6.0f } };
    ctx.dl->rect_filled(track, ctx.apply(ctx.theme->control), 3.0f);

    float t = 0.0f;
    if (value_ != nullptr && max_ > min_) t = clamp01((*value_ - min_) / (max_ - min_));

    const Rect fill{ track.pos, { track.width() * t, track.height() } };
    ctx.dl->rect_filled(fill, ctx.apply(ctx.theme->accent), 3.0f);

    const float hx = rect_.left() + rect_.width() * t;
    const Rect  knob{ { hx - 4.0f, rect_.top() }, { 8.0f, rect_.height() } };
    const Color kc = ctx.ui->is_active(this) ? ctx.theme->control_active : ctx.theme->control_hot;
    ctx.dl->rect_filled(knob, ctx.apply(kc), 2.0f);
}

// -----------------------------------------------------------------------------
// Canvas
// -----------------------------------------------------------------------------
Widget* Canvas::add_anchored(cardinal::unique_ptr<Widget> child, Anchor a, Vec2 offset) {
    Widget* raw = add(cardinal::move(child));
    slots_.push_back(Slot{ a, offset });
    return raw;
}

Vec2 Canvas::measure(const Constraints& c) {
    // A positioning container fills the space it is offered; children are placed
    // within that, they don't grow it. (Measure children so they're sized.)
    for (auto& ch : children_)
        if (ch && ch->visible()) ch->measure(Constraints::loose(
            c.max_w < 1e8f ? c.max_w : 0.0f, c.max_h < 1e8f ? c.max_h : 0.0f));
    Vec2 sz{ c.clamp_w(c.max_w < 1e8f ? c.max_w : 0.0f),
             c.clamp_h(c.max_h < 1e8f ? c.max_h : 0.0f) };
    measured_ = sz;
    return sz;
}

void Canvas::arrange(const Rect& r) {
    rect_ = r;
    for (cardinal::usize i = 0; i < children_.size(); ++i) {
        auto& ch = children_[i];
        if (!ch || !ch->visible()) continue;
        const Slot slot = (i < slots_.size()) ? slots_[i] : Slot{};
        const Vec2 s = ch->measure(Constraints::loose(r.width(), r.height()));

        const int col = anchor_col(slot.anchor);   // 0 L / 1 C / 2 R
        const int row = anchor_row(slot.anchor);    // 0 T / 1 M / 2 B
        float x;
        if      (col == 0) x = r.left()  + slot.offset.x;                       // inset from left
        else if (col == 1) x = r.left()  + (r.width()  - s.x) * 0.5f + slot.offset.x;
        else               x = r.right() - s.x - slot.offset.x;                  // inset from right
        float y;
        if      (row == 0) y = r.top()    + slot.offset.y;                       // inset from top
        else if (row == 1) y = r.top()    + (r.height() - s.y) * 0.5f + slot.offset.y;
        else               y = r.bottom() - s.y - slot.offset.y;                 // inset from bottom

        ch->arrange(Rect{ { x, y }, s });
    }
}

// -----------------------------------------------------------------------------
// ScrollPanel
// -----------------------------------------------------------------------------
Vec2 ScrollPanel::measure(const Constraints& c) {
    // A viewport fills the space it's offered; content scrolls within.
    Vec2 sz{ c.clamp_w(c.max_w < 1e8f ? c.max_w : 0.0f),
             c.clamp_h(c.max_h < 1e8f ? c.max_h : 0.0f) };
    measured_ = sz;
    return sz;
}

void ScrollPanel::arrange(const Rect& r) {
    rect_ = r;

    // Content height = stacked children + spacing.
    float total = 0.0f;
    int n = 0;
    for (auto& ch : children_) {
        if (!ch || !ch->visible()) continue;
        total += ch->measure(Constraints::loose(r.width(), 1.0e9f)).y + spacing_;
        ++n;
    }
    if (n > 0) total -= spacing_;
    content_h_ = total;

    // Clamp scroll to [0, content - viewport].
    float maxs = content_h_ - r.height();
    if (maxs < 0.0f) maxs = 0.0f;
    if (scroll_ > maxs) scroll_ = maxs;
    if (scroll_ < 0.0f) scroll_ = 0.0f;

    // Place children top-to-bottom, shifted up by the scroll offset.
    float y = r.top() - scroll_;
    for (auto& ch : children_) {
        if (!ch || !ch->visible()) continue;
        const Vec2 s = ch->measure(Constraints::loose(r.width(), 1.0e9f));
        ch->arrange(Rect{ { r.left(), y }, { r.width(), s.y } });
        y += s.y + spacing_;
    }
}

void ScrollPanel::on_scroll(float delta) {
    scroll_ -= delta * 30.0f;      // wheel up -> toward the top; clamped in arrange
    if (scroll_ < 0.0f) scroll_ = 0.0f;
}

void ScrollPanel::paint(PaintContext& ctx) {
    ctx.dl->push_clip(rect_);
    paint_children(ctx);
    ctx.dl->pop_clip();

    // A thin scrollbar when the content overflows.
    if (content_h_ > rect_.height() && rect_.height() > 0.0f) {
        const float frac  = rect_.height() / content_h_;
        const float bar_h = rect_.height() * frac;
        const float maxs  = content_h_ - rect_.height();
        const float t     = maxs > 0.0f ? scroll_ / maxs : 0.0f;
        const float bar_y = rect_.top() + t * (rect_.height() - bar_h);
        const Rect  bar{ { rect_.right() - 4.0f, bar_y }, { 3.0f, bar_h } };
        ctx.dl->rect_filled(bar, ctx.apply(ctx.theme->control_hot), 1.5f);
    }
}

// -----------------------------------------------------------------------------
// TextField
// -----------------------------------------------------------------------------
Vec2 TextField::measure(const Constraints& c) {
    Vec2 s{ maxf(min_width_, text_advance(text_, font_size_) + pad_.horiz()),
            line_height(font_size_) + pad_.vert() };
    s.x = c.clamp_w(s.x);
    s.y = c.clamp_h(s.y);
    measured_ = s;
    return s;
}

void TextField::on_key(const InputState& in) {
    const cardinal::string before = text_;

    // Printable insert at the caret (ASCII subset; hosts pre-filter).
    for (char ch : in.text) {
        if (static_cast<unsigned char>(ch) < 32 || ch == 127) continue;
        text_.insert(text_.begin() + static_cast<isize>(caret_), ch);
        ++caret_;
    }
    if (in.key_backspace && caret_ > 0) {
        text_.erase(text_.begin() + static_cast<isize>(caret_) - 1);
        --caret_;
    }
    if (in.key_delete && caret_ < text_.size()) {
        text_.erase(text_.begin() + static_cast<isize>(caret_));
    }
    if (in.key_left  && caret_ > 0)            --caret_;
    if (in.key_right && caret_ < text_.size()) ++caret_;
    if (in.key_home) caret_ = 0;
    if (in.key_end)  caret_ = text_.size();

    if (text_ != before && on_change_) on_change_(text_);
    if (in.key_enter && on_submit_)    on_submit_(text_);
}

void TextField::paint(PaintContext& ctx) {
    const bool focused = ctx.ui != nullptr && ctx.ui->is_focused(this);

    ctx.dl->rect_filled(rect_, ctx.apply(ctx.theme->control), ctx.theme->rounding);
    ctx.dl->rect_stroke(rect_,
                        ctx.apply(focused ? ctx.theme->accent : ctx.theme->border),
                        1.0f, ctx.theme->rounding);

    const Vec2 tp{ rect_.left() + pad_.l,
                   rect_.top()  + (rect_.height() - line_height(font_size_)) * 0.5f };
    if (!text_.empty()) {
        ctx.dl->text(tp, text_, ctx.apply(ctx.theme->text), font_size_);
    } else if (!placeholder_.empty()) {
        ctx.dl->text(tp, placeholder_, ctx.apply(ctx.theme->text_dim), font_size_);
    }

    // Caret (solid while focused) after the first `caret_` characters.
    if (focused) {
        const cardinal::string head = text_.substr(0, caret_);
        const float cx = tp.x + text_advance(head, font_size_);
        const Rect  caret{ { cx, tp.y }, { 1.0f, line_height(font_size_) } };
        ctx.dl->rect_filled(caret, ctx.apply(ctx.theme->text), 0.0f);
    }
}

}  // namespace cardinal::cui
