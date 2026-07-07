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

// -----------------------------------------------------------------------------
// ListView
// -----------------------------------------------------------------------------
Vec2 ListView::measure(const Constraints& c) {
    float w = 0.0f;
    for (const auto& it : items_) w = maxf(w, text_advance(it, font_size_));
    Vec2 s{ w + pad_.horiz(), row_h_ * static_cast<float>(items_.size()) };
    s.x = c.clamp_w(s.x);
    s.y = c.clamp_h(s.y);
    measured_ = s;
    return s;
}

int ListView::row_at(Vec2 pt) const noexcept {
    if (!rect_.contains(pt) || row_h_ <= 0.0f) return -1;
    const int i = static_cast<int>((pt.y - rect_.top()) / row_h_);
    return (i >= 0 && i < static_cast<int>(items_.size())) ? i : -1;
}

void ListView::on_drag(Vec2 mouse) {
    const int i = row_at(mouse);
    if (i >= 0 && i != selected_) {
        selected_ = i;
        if (on_select_) on_select_(i);
    }
}

void ListView::paint(PaintContext& ctx) {
    float y = rect_.top();
    for (int i = 0; i < static_cast<int>(items_.size()); ++i) {
        const Rect row{ { rect_.left(), y }, { rect_.width(), row_h_ } };
        if (i == selected_)
            ctx.dl->rect_filled(row, ctx.apply(ctx.theme->control_active), 2.0f);
        const Vec2 tp{ row.left() + pad_.l,
                       row.top()  + (row_h_ - line_height(font_size_)) * 0.5f };
        ctx.dl->text(tp, items_[i], ctx.apply(ctx.theme->text), font_size_);
        y += row_h_;
    }
}

// -----------------------------------------------------------------------------
// TreeView
// -----------------------------------------------------------------------------
int TreeView::add_node(int parent, cardinal::string label, bool expanded) {
    const int h = static_cast<int>(nodes_.size());
    Node n;
    n.label    = cardinal::move(label);
    n.parent   = (parent >= 0 && parent < h) ? parent : -1;
    n.expanded = expanded;
    nodes_.push_back(cardinal::move(n));
    if (nodes_[h].parent >= 0) nodes_[nodes_[h].parent].children.push_back(h);
    else                       roots_.push_back(h);
    return h;
}

void TreeView::flatten(cardinal::vector<Row>& out) const {
    // Iterative DFS over the visible (expanded) tree, in insertion order.
    cardinal::vector<Row> stack;
    for (usize r = roots_.size(); r > 0; --r)
        stack.push_back({ roots_[r - 1], 0 });
    while (!stack.empty()) {
        const Row cur = stack.back();
        stack.pop_back();
        out.push_back(cur);
        const Node& n = nodes_[cur.node];
        if (n.expanded)
            for (usize c = n.children.size(); c > 0; --c)
                stack.push_back({ n.children[c - 1], cur.depth + 1 });
    }
}

Vec2 TreeView::measure(const Constraints& c) {
    cardinal::vector<Row> rows;
    flatten(rows);
    float w = 0.0f;
    for (const auto& r : rows)
        w = maxf(w, static_cast<float>(r.depth) * indent_ + arrow_w_ +
                    text_advance(nodes_[r.node].label, font_size_));
    Vec2 s{ w + pad_.horiz(), row_h_ * static_cast<float>(rows.size()) };
    s.x = c.clamp_w(s.x);
    s.y = c.clamp_h(s.y);
    measured_ = s;
    return s;
}

void TreeView::on_click() {
    if (!press_valid_) return;
    press_valid_ = false;
    cardinal::vector<Row> rows;
    flatten(rows);
    if (!rect_.contains(press_) || row_h_ <= 0.0f) return;
    const int i = static_cast<int>((press_.y - rect_.top()) / row_h_);
    if (i < 0 || i >= static_cast<int>(rows.size())) return;
    const Row&  row = rows[static_cast<usize>(i)];
    const Node& n   = nodes_[row.node];
    const float arrow_x0 = rect_.left() + pad_.l +
                           static_cast<float>(row.depth) * indent_;
    if (!n.children.empty() && press_.x < arrow_x0 + arrow_w_) {
        nodes_[row.node].expanded = !n.expanded;   // arrow region -> toggle
        return;
    }
    if (selected_ != row.node) {
        selected_ = row.node;
        if (on_select_) on_select_(row.node);
    }
}

void TreeView::paint(PaintContext& ctx) {
    cardinal::vector<Row> rows;
    flatten(rows);
    float y = rect_.top();
    for (const auto& r : rows) {
        const Node& n = nodes_[r.node];
        const Rect row{ { rect_.left(), y }, { rect_.width(), row_h_ } };
        if (r.node == selected_)
            ctx.dl->rect_filled(row, ctx.apply(ctx.theme->control_active), 2.0f);
        const float x0 = rect_.left() + pad_.l + static_cast<float>(r.depth) * indent_;
        const float ty = row.top() + (row_h_ - line_height(font_size_)) * 0.5f;
        if (!n.children.empty())
            ctx.dl->text({ x0, ty }, n.expanded ? "v" : ">",
                         ctx.apply(ctx.theme->text_dim), font_size_);
        ctx.dl->text({ x0 + arrow_w_, ty }, n.label,
                     ctx.apply(ctx.theme->text), font_size_);
        y += row_h_;
    }
}

// -----------------------------------------------------------------------------
// Combo
// -----------------------------------------------------------------------------
float Combo::header_h() const noexcept {
    return line_height(font_size_) + pad_.vert();
}

Vec2 Combo::measure(const Constraints& c) {
    float w = 0.0f;
    for (const auto& it : items_) w = maxf(w, text_advance(it, font_size_));
    w = maxf(w, text_advance(placeholder_, font_size_));
    Vec2 s{ maxf(min_width_, w + pad_.horiz() + arrow_w_), header_h() };
    s.x = c.clamp_w(s.x);
    s.y = c.clamp_h(s.y);
    measured_ = s;
    return s;
}

void Combo::arrange(const Rect& r) {
    // Intrinsic header height (a stretching parent must not inflate the
    // closed control); while open, the rect extends over the dropdown so the
    // rows stay inside this widget's hit region.
    float h = header_h();
    if (open_) h += row_h_ * static_cast<float>(items_.size());
    rect_ = { r.pos, { r.width(), h } };
}

void Combo::on_click() {
    if (!press_valid_) return;
    press_valid_ = false;
    const float drop_top = rect_.top() + header_h();
    if (press_.y < drop_top) {                       // header -> toggle
        open_ = !open_;
        return;
    }
    if (!open_ || row_h_ <= 0.0f) return;            // dropdown row -> select
    const int i = static_cast<int>((press_.y - drop_top) / row_h_);
    if (i >= 0 && i < static_cast<int>(items_.size())) {
        if (i != selected_) {
            selected_ = i;
            if (on_select_) on_select_(i);
        }
        open_ = false;
    }
}

void Combo::paint(PaintContext& ctx) {
    const Rect header{ rect_.pos, { rect_.width(), header_h() } };

    Color bg = ctx.theme->control;
    if      (open_)                    bg = ctx.theme->control_active;
    else if (ctx.ui->is_hovered(this)) bg = ctx.theme->control_hot;
    ctx.dl->rect_filled(header, ctx.apply(bg), ctx.theme->rounding);
    ctx.dl->rect_stroke(header,
                        ctx.apply(ctx.ui->is_focused(this) ? ctx.theme->accent
                                                           : ctx.theme->border),
                        1.0f, ctx.theme->rounding);

    const float ty = header.top() + (header.height() - line_height(font_size_)) * 0.5f;
    if (selected_ >= 0 && selected_ < static_cast<int>(items_.size()))
        ctx.dl->text({ header.left() + pad_.l, ty },
                     items_[static_cast<usize>(selected_)],
                     ctx.apply(ctx.theme->text), font_size_);
    else
        ctx.dl->text({ header.left() + pad_.l, ty }, placeholder_,
                     ctx.apply(ctx.theme->text_dim), font_size_);
    ctx.dl->text({ header.right() - arrow_w_ + 2.0f, ty }, open_ ? "^" : "v",
                 ctx.apply(ctx.theme->text_dim), font_size_);

    if (!open_) return;
    const Rect drop{ { rect_.left(), rect_.top() + header_h() },
                     { rect_.width(), row_h_ * static_cast<float>(items_.size()) } };
    ctx.dl->rect_filled(drop, ctx.apply(ctx.theme->panel_bg), ctx.theme->rounding);
    ctx.dl->rect_stroke(drop, ctx.apply(ctx.theme->border), 1.0f, ctx.theme->rounding);
    float y = drop.top();
    for (int i = 0; i < static_cast<int>(items_.size()); ++i) {
        const Rect row{ { drop.left(), y }, { drop.width(), row_h_ } };
        if (i == selected_)
            ctx.dl->rect_filled(row, ctx.apply(ctx.theme->control_active), 2.0f);
        ctx.dl->text({ row.left() + pad_.l,
                       row.top() + (row_h_ - line_height(font_size_)) * 0.5f },
                     items_[static_cast<usize>(i)],
                     ctx.apply(ctx.theme->text), font_size_);
        y += row_h_;
    }
}

// -----------------------------------------------------------------------------
// MenuBar
// -----------------------------------------------------------------------------
float MenuBar::title_w(int i) const noexcept {
    return text_advance(menus_[static_cast<usize>(i)].title, font_size_) +
           2.0f * title_pad_;
}

float MenuBar::title_x(int i) const noexcept {
    float x = 0.0f;
    for (int m = 0; m < i; ++m) x += title_w(m);
    return x;
}

Rect MenuBar::popup_rect() const noexcept {
    const Menu& m = menus_[static_cast<usize>(open_)];
    float w = 0.0f;
    for (const auto& it : m.items) w = maxf(w, text_advance(it, font_size_));
    w = maxf(w + 2.0f * item_pad_, title_w(open_));
    return { { rect_.left() + title_x(open_), rect_.top() + bar_h_ },
             { w, row_h_ * static_cast<float>(m.items.size()) } };
}

Vec2 MenuBar::measure(const Constraints& c) {
    float titles = 0.0f;
    for (int i = 0; i < static_cast<int>(menus_.size()); ++i) titles += title_w(i);
    Vec2 s{ (c.max_w < 1e8f) ? c.max_w : titles, bar_h_ };
    s.x = c.clamp_w(s.x);
    s.y = c.clamp_h(s.y);
    measured_ = s;
    return s;
}

void MenuBar::arrange(const Rect& r) {
    float h = bar_h_;
    if (open_ >= 0 && open_ < static_cast<int>(menus_.size()))
        h += row_h_ * static_cast<float>(menus_[static_cast<usize>(open_)].items.size());
    rect_ = { r.pos, { r.width(), h } };
}

void MenuBar::on_click() {
    if (!press_valid_) return;
    press_valid_ = false;

    if (press_.y < rect_.top() + bar_h_) {           // bar -> toggle/switch menu
        const float rel = press_.x - rect_.left();
        float x = 0.0f;
        for (int i = 0; i < static_cast<int>(menus_.size()); ++i) {
            const float w = title_w(i);
            if (rel >= x && rel < x + w) {
                open_ = (open_ == i) ? -1 : i;
                return;
            }
            x += w;
        }
        open_ = -1;                                   // bar, right of all titles
        return;
    }

    if (open_ < 0) return;
    const Rect pr = popup_rect();                     // popup row -> fire item
    if (pr.contains(press_) && row_h_ > 0.0f) {
        const int i = static_cast<int>((press_.y - pr.top()) / row_h_);
        const int menu = open_;
        open_ = -1;
        if (i >= 0 &&
            i < static_cast<int>(menus_[static_cast<usize>(menu)].items.size()) &&
            on_item_)
            on_item_(menu, i);
        return;
    }
    open_ = -1;                                       // stray click in the band
}

void MenuBar::paint(PaintContext& ctx) {
    const Rect bar{ rect_.pos, { rect_.width(), bar_h_ } };
    ctx.dl->rect_filled(bar, ctx.apply(ctx.theme->control), 0.0f);

    float x = bar.left();
    const float ty = bar.top() + (bar_h_ - line_height(font_size_)) * 0.5f;
    for (int i = 0; i < static_cast<int>(menus_.size()); ++i) {
        const float w = title_w(i);
        if (i == open_)
            ctx.dl->rect_filled({ { x, bar.top() }, { w, bar_h_ } },
                                ctx.apply(ctx.theme->control_active), 0.0f);
        ctx.dl->text({ x + title_pad_, ty },
                     menus_[static_cast<usize>(i)].title,
                     ctx.apply(ctx.theme->text), font_size_);
        x += w;
    }

    if (open_ < 0) return;
    const Rect pr = popup_rect();
    ctx.dl->rect_filled(pr, ctx.apply(ctx.theme->panel_bg), ctx.theme->rounding);
    ctx.dl->rect_stroke(pr, ctx.apply(ctx.theme->border), 1.0f, ctx.theme->rounding);
    const Menu& m = menus_[static_cast<usize>(open_)];
    float y = pr.top();
    for (const auto& it : m.items) {
        ctx.dl->text({ pr.left() + item_pad_,
                       y + (row_h_ - line_height(font_size_)) * 0.5f },
                     it, ctx.apply(ctx.theme->text), font_size_);
        y += row_h_;
    }
}

// -----------------------------------------------------------------------------
// Tabs
// -----------------------------------------------------------------------------
Widget* Tabs::add_tab(cardinal::string label, cardinal::unique_ptr<Widget> content) {
    labels_.push_back(cardinal::move(label));
    Widget* raw = add(cardinal::move(content));
    raw->set_visible(static_cast<int>(labels_.size()) - 1 == selected_);
    return raw;
}

float Tabs::tab_w(int i) const noexcept {
    return text_advance(labels_[static_cast<usize>(i)], font_size_) + 2.0f * tab_pad_;
}

void Tabs::set_selected(int i) {
    if (i < 0 || i >= count() || i == selected_) return;
    selected_ = i;
    for (int c = 0; c < static_cast<int>(children_.size()); ++c)
        if (children_[static_cast<usize>(c)])
            children_[static_cast<usize>(c)]->set_visible(c == selected_);
    // Arrange the newly shown page immediately (its rect may be stale/empty).
    if (rect_.width() > 0.0f && selected_ < static_cast<int>(children_.size())) {
        auto& ch = children_[static_cast<usize>(selected_)];
        if (ch) {
            const Rect cr = content_rect();
            ch->measure(Constraints::loose(cr.width(), cr.height()));
            ch->arrange(cr);
        }
    }
}

Vec2 Tabs::measure(const Constraints& c) {
    // A page host fills the space it is offered (headers + content region).
    float titles = 0.0f;
    for (int i = 0; i < count(); ++i) titles += tab_w(i);
    const Constraints inner = Constraints::loose(
        c.max_w < 1e8f ? c.max_w : titles,
        c.max_h < 1e8f ? (c.max_h > tab_h_ ? c.max_h - tab_h_ : 0.0f) : 0.0f);
    for (auto& ch : children_)
        if (ch && ch->visible()) ch->measure(inner);
    Vec2 s{ c.clamp_w(c.max_w < 1e8f ? c.max_w : titles),
            c.clamp_h(c.max_h < 1e8f ? c.max_h : tab_h_) };
    measured_ = s;
    return s;
}

void Tabs::arrange(const Rect& r) {
    rect_ = r;
    const Rect cr = content_rect();
    for (int i = 0; i < static_cast<int>(children_.size()); ++i) {
        auto& ch = children_[static_cast<usize>(i)];
        if (!ch) continue;
        ch->set_visible(i == selected_);
        if (i == selected_) {
            ch->measure(Constraints::loose(cr.width(), cr.height()));
            ch->arrange(cr);
        }
    }
}

void Tabs::on_click() {
    if (!press_valid_) return;
    press_valid_ = false;
    if (press_.y >= rect_.top() + tab_h_) return;     // content area -> not ours
    const float rel = press_.x - rect_.left();
    float x = 0.0f;
    for (int i = 0; i < count(); ++i) {
        const float w = tab_w(i);
        if (rel >= x && rel < x + w) {
            if (i != selected_) {
                set_selected(i);
                if (on_change_) on_change_(i);
            }
            return;
        }
        x += w;
    }
}

void Tabs::paint(PaintContext& ctx) {
    const Rect strip{ rect_.pos, { rect_.width(), tab_h_ } };
    ctx.dl->rect_filled(strip, ctx.apply(ctx.theme->window_bg), 0.0f);

    float x = strip.left();
    const float ty = strip.top() + (tab_h_ - line_height(font_size_)) * 0.5f;
    for (int i = 0; i < count(); ++i) {
        const float w = tab_w(i);
        const Rect tab{ { x, strip.top() }, { w, tab_h_ } };
        ctx.dl->rect_filled(tab,
                            ctx.apply(i == selected_ ? ctx.theme->panel_bg
                                                     : ctx.theme->control),
                            ctx.theme->rounding);
        if (i == selected_)                            // accent underline
            ctx.dl->rect_filled({ { x, strip.bottom() - 2.0f }, { w, 2.0f } },
                                ctx.apply(ctx.theme->accent), 0.0f);
        ctx.dl->text({ x + tab_pad_, ty }, labels_[static_cast<usize>(i)],
                     ctx.apply(i == selected_ ? ctx.theme->text : ctx.theme->text_dim),
                     font_size_);
        x += w;
    }

    const Rect cr = content_rect();
    ctx.dl->rect_filled(cr, ctx.apply(ctx.theme->panel_bg), 0.0f);
    ctx.dl->rect_stroke(rect_, ctx.apply(ctx.theme->border), 1.0f, 0.0f);
    paint_children(ctx);                               // only the visible page paints
}

}  // namespace cardinal::cui
