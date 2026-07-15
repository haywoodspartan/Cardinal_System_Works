#pragma once

// =============================================================================
// Cardinal UI — core widget set (MVP).
//
//   Stack    — arranges children along an axis (V/H) with spacing + alignment
//   Panel    — a filled, padded, bordered container (usually wraps a Stack)
//   Label    — static text
//   Button   — clickable text button (hover/active states + click callback)
//   Checkbox — bound bool toggle with a label
//   Slider   — bound float in [min,max] with a draggable knob
//
// Later phases added: ScrollPanel, TextField, ListView, TreeView, Combo,
// MenuBar, Tabs. Still to come: dock, table, multiline text.
// =============================================================================

#include <cardinal/core/types.hpp>

#include "widget.hpp"

namespace cardinal::cui {

// Linear container. Children are measured at their natural size; FLEX
// children (add_flex) instead share the leftover main-axis space by weight —
// the missing primitive for panel layouts like "scrollback fills everything
// above the input row". Weights ride a parallel vector (same pattern as
// Canvas slots_); plain add() children default to weight 0 (natural size).
class Stack : public Widget {
public:
    explicit Stack(Axis axis, float spacing = 4.0f, Align cross = Align::Stretch)
        : axis_(axis), spacing_(spacing), cross_(cross) {}
    // Add `child` with a flex-grow weight (> 0). It receives
    // leftover * weight / total_weight of the main axis at arrange time.
    Widget* add_flex(cardinal::unique_ptr<Widget> child, float weight);
    Vec2 measure(const Constraints& c) override;
    void arrange(const Rect& r) override;
    void paint(PaintContext& ctx) override { paint_children(ctx); }
private:
    float weight_of(usize i) const noexcept {
        return (i < flex_.size()) ? flex_[i] : 0.0f;
    }
    Axis  axis_;
    float spacing_;
    Align cross_;
    cardinal::vector<float> flex_;   // parallel to children_ (missing = 0)
};

// Filled, padded, bordered container (single logical child — usually a Stack).
class Panel : public Widget {
public:
    explicit Panel(Color bg, Edges pad = Edges::all(6.0f)) : bg_(bg), pad_(pad) {}
    Vec2 measure(const Constraints& c) override;
    void arrange(const Rect& r) override;
    void paint(PaintContext& ctx) override;
private:
    Color bg_;
    Edges pad_;
};

class Label : public Widget {
public:
    explicit Label(cardinal::string text) : text_(cardinal::move(text)) {}
    Vec2 measure(const Constraints& c) override;
    void paint(PaintContext& ctx) override;
    void set_text(cardinal::string t) { text_ = cardinal::move(t); }
    const cardinal::string& text() const noexcept { return text_; }
private:
    cardinal::string text_;
    float font_size_{14.0f};
};

class Button : public Widget {
public:
    explicit Button(cardinal::string text, cardinal::function<void()> on_click = {})
        : text_(cardinal::move(text)), on_click_(cardinal::move(on_click)) {}
    Vec2 measure(const Constraints& c) override;
    void paint(PaintContext& ctx) override;
    bool interactive() const noexcept override { return true; }
    void on_click() override { if (on_click_) on_click_(); }
private:
    cardinal::string           text_;
    cardinal::function<void()> on_click_;
    float font_size_{14.0f};
    Edges pad_{8, 4, 8, 4};
};

class Checkbox : public Widget {
public:
    Checkbox(cardinal::string label, bool* value)
        : label_(cardinal::move(label)), value_(value) {}
    Vec2 measure(const Constraints& c) override;
    void paint(PaintContext& ctx) override;
    bool interactive() const noexcept override { return true; }
    void on_click() override { if (value_) *value_ = !*value_; }
private:
    cardinal::string label_;
    bool* value_{nullptr};
    float font_size_{14.0f};
    float box_{16.0f};
    float gap_{6.0f};
};

class Slider : public Widget {
public:
    Slider(float* value, float min_v, float max_v)
        : value_(value), min_(min_v), max_(max_v) {}
    Vec2 measure(const Constraints& c) override;
    void paint(PaintContext& ctx) override;
    bool interactive() const noexcept override { return true; }
    void on_drag(Vec2 mouse) override;
private:
    float* value_{nullptr};
    float  min_{0.0f}, max_{1.0f};
    float  height_{20.0f};
};

// Canvas — absolute/anchor positioning container. Each child is placed within
// the canvas's rect by a 9-point Anchor + a pixel offset (a "pan" inset from
// the anchored edge), keeping its own measured size. This is Cardinal's clean
// take on Pearl Abyss's PAUIBase anchor model (ComputePos): game UIs need
// anchored absolute layout, not just flow. Best used as a root / full-region
// container (it fills the space it is offered).
class Canvas : public Widget {
public:
    Canvas() = default;
    // Add `child` anchored at `a`, offset inward from the anchored edge.
    Widget* add_anchored(cardinal::unique_ptr<Widget> child, Anchor a, Vec2 offset = {});
    Vec2 measure(const Constraints& c) override;
    void arrange(const Rect& r) override;
    void paint(PaintContext& ctx) override { paint_children(ctx); }
private:
    struct Slot { Anchor anchor{Anchor::TopLeft}; Vec2 offset{}; };
    cardinal::vector<Slot> slots_;   // parallel to children_
};

// ScrollPanel — vertical scroll viewport. Stacks children top-to-bottom, clips
// them to its rect, and offsets by a scroll position the mouse wheel drives.
// The editor's most-needed container (panels are long lists). Clipping rides
// the DrawList clip stack -> the ImGui bridge applies it as a scissor.
class ScrollPanel : public Widget {
public:
    explicit ScrollPanel(float spacing = 4.0f) : spacing_(spacing) {}
    Vec2 measure(const Constraints& c) override;
    void arrange(const Rect& r) override;
    void paint(PaintContext& ctx) override;
    bool scrollable() const noexcept override { return true; }
    void on_scroll(float delta) override;
    float scroll() const noexcept { return scroll_; }
    // Pin the view to the bottom (consoles/logs after an append). The huge
    // value is clamped to the real max in the next arrange().
    void scroll_to_end() noexcept { scroll_ = 1.0e9f; }
private:
    float spacing_;
    float scroll_{0.0f};      // how far down the content is scrolled (>= 0)
    float content_h_{0.0f};   // total content height (for clamping + scrollbar)
};

// TextField — single-line text input. Click to focus (the Ui routes keyboard
// input here while focused); supports insert-at-caret, Backspace/Delete,
// Left/Right/Home/End, Enter -> on_submit, Escape -> unfocus (Ui-consumed).
// The editor's gating widget: Console, search boxes, Inspector name field.
class TextField : public Widget {
public:
    explicit TextField(cardinal::string text = {}, float min_width = 160.0f)
        : text_(cardinal::move(text)), caret_(text_.size()), min_width_(min_width) {}

    Vec2 measure(const Constraints& c) override;
    void paint(PaintContext& ctx) override;
    bool interactive() const noexcept override { return true; }
    bool focusable()   const noexcept override { return true; }
    void on_click() override { caret_ = text_.size(); }   // caret -> end (for now)
    void on_key(const InputState& in) override;

    const cardinal::string& text() const noexcept { return text_; }
    void set_text(cardinal::string t) {
        text_ = cardinal::move(t);
        if (caret_ > text_.size()) caret_ = text_.size();
    }
    usize caret() const noexcept { return caret_; }
    void  set_caret(usize c) noexcept { caret_ = c > text_.size() ? text_.size() : c; }
    void set_placeholder(cardinal::string p) { placeholder_ = cardinal::move(p); }
    void set_on_change(cardinal::function<void(const cardinal::string&)> f) {
        on_change_ = cardinal::move(f);
    }
    void set_on_submit(cardinal::function<void(const cardinal::string&)> f) {
        on_submit_ = cardinal::move(f);
    }

private:
    cardinal::string text_;
    cardinal::string placeholder_;
    usize caret_{0};
    float min_width_{160.0f};
    float font_size_{14.0f};
    Edges pad_{6, 4, 6, 4};
    cardinal::function<void(const cardinal::string&)> on_change_;
    cardinal::function<void(const cardinal::string&)> on_submit_;
};

// ListView — selectable flat list of text rows. Selection follows the press
// (and drag-through, UE-style); on_select fires on change. Not a scroller —
// wrap in a ScrollPanel for long lists (composition).
class ListView : public Widget {
public:
    ListView() = default;
    void set_items(cardinal::vector<cardinal::string> items) {
        items_ = cardinal::move(items);
        if (selected_ >= static_cast<int>(items_.size())) selected_ = -1;
    }
    void add_item(cardinal::string item) { items_.push_back(cardinal::move(item)); }
    void clear_items() { items_.clear(); selected_ = -1; }
    const cardinal::vector<cardinal::string>& items() const noexcept { return items_; }

    int  selected() const noexcept { return selected_; }
    void set_selected(int i) {
        selected_ = (i >= 0 && i < static_cast<int>(items_.size())) ? i : -1;
    }
    void set_on_select(cardinal::function<void(int)> f) { on_select_ = cardinal::move(f); }

    Vec2 measure(const Constraints& c) override;
    void paint(PaintContext& ctx) override;
    bool interactive() const noexcept override { return true; }
    void on_drag(Vec2 mouse) override;    // press + drag-through selection

private:
    int row_at(Vec2 pt) const noexcept;
    cardinal::vector<cardinal::string> items_;
    int   selected_{-1};
    float font_size_{14.0f};
    float row_h_{22.0f};
    Edges pad_{6, 3, 6, 3};
    cardinal::function<void(int)> on_select_;
};

// TreeView — hierarchy with expand/collapse (the Outliner's widget). Nodes
// live in an internal pool; add_node(parent, label) returns a stable handle
// (-1 = top level). Clicking a row's arrow region toggles expansion; clicking
// the label selects (fires on_select with the node handle, once per click).
class TreeView : public Widget {
public:
    TreeView() = default;
    int add_node(int parent, cardinal::string label, bool expanded = true);
    void clear() { nodes_.clear(); roots_.clear(); selected_ = -1; }

    int  selected() const noexcept { return selected_; }
    void set_selected(int h) {
        selected_ = (h >= 0 && h < static_cast<int>(nodes_.size())) ? h : -1;
    }
    void set_on_select(cardinal::function<void(int)> f) { on_select_ = cardinal::move(f); }
    bool expanded(int h) const {
        return h >= 0 && h < static_cast<int>(nodes_.size()) && nodes_[h].expanded;
    }
    void set_expanded(int h, bool e) {
        if (h >= 0 && h < static_cast<int>(nodes_.size())) nodes_[h].expanded = e;
    }
    const cardinal::string& label(int h) const { return nodes_[h].label; }

    Vec2 measure(const Constraints& c) override;
    void paint(PaintContext& ctx) override;
    bool interactive() const noexcept override { return true; }
    void on_drag(Vec2 mouse) override { press_ = mouse; press_valid_ = true; }
    void on_click() override;             // toggle arrow / select label

private:
    struct Node {
        cardinal::string      label;
        int                   parent{-1};
        cardinal::vector<int> children;
        bool                  expanded{true};
    };
    struct Row { int node; int depth; };
    void flatten(cardinal::vector<Row>& out) const;

    cardinal::vector<Node> nodes_;
    cardinal::vector<int>  roots_;
    int   selected_{-1};
    float font_size_{14.0f};
    float row_h_{22.0f};
    float indent_{16.0f};
    float arrow_w_{14.0f};
    Edges pad_{4, 3, 6, 3};
    Vec2  press_{};
    bool  press_valid_{false};
    cardinal::function<void(int)> on_select_;
};

// Combo — dropdown selector. Closed it's a one-line control (current item +
// arrow); clicking opens an item list BELOW it. While open the widget extends
// its own rect over the dropdown so the rows stay hit-testable; hosts should
// hand it to Ui::set_popup while open so it also wins input/paint over later
// siblings. Click-away and Escape close via focus loss (focusable widget).
class Combo : public Widget {
public:
    Combo() = default;
    void set_items(cardinal::vector<cardinal::string> items) {
        items_ = cardinal::move(items);
        if (selected_ >= static_cast<int>(items_.size())) selected_ = -1;
    }
    void add_item(cardinal::string item) { items_.push_back(cardinal::move(item)); }
    const cardinal::vector<cardinal::string>& items() const noexcept { return items_; }

    int  selected() const noexcept { return selected_; }
    void set_selected(int i) {
        selected_ = (i >= 0 && i < static_cast<int>(items_.size())) ? i : -1;
    }
    void set_on_select(cardinal::function<void(int)> f) { on_select_ = cardinal::move(f); }
    void set_placeholder(cardinal::string p) { placeholder_ = cardinal::move(p); }
    bool open() const noexcept { return open_; }

    Vec2 measure(const Constraints& c) override;
    void arrange(const Rect& r) override;      // intrinsic header height + dropdown
    void paint(PaintContext& ctx) override;
    bool interactive() const noexcept override { return true; }
    bool focusable()   const noexcept override { return true; }
    void on_drag(Vec2 mouse) override { press_ = mouse; press_valid_ = true; }
    void on_click() override;
    void on_focus_changed(bool focused) override { if (!focused) open_ = false; }

private:
    float header_h() const noexcept;
    cardinal::vector<cardinal::string> items_;
    cardinal::string placeholder_{"Select..."};
    int   selected_{-1};
    bool  open_{false};
    float min_width_{160.0f};
    float font_size_{14.0f};
    float row_h_{22.0f};
    float arrow_w_{16.0f};
    Edges pad_{8, 4, 8, 4};
    Vec2  press_{};
    bool  press_valid_{false};
    cardinal::function<void(int)> on_select_;
};

// MenuBar — horizontal bar of menu titles; clicking a title drops its item
// list below the bar. add_menu returns a menu handle; on_item fires with
// (menu, item). Same popup contract as Combo: extends rect_ while open, hosts
// pass it to Ui::set_popup, click-away/Escape close via focus loss. A stray
// click inside the extended region but outside the popup just closes it.
class MenuBar : public Widget {
public:
    MenuBar() = default;
    int  add_menu(cardinal::string title) {
        menus_.push_back(Menu{ cardinal::move(title), {} });
        return static_cast<int>(menus_.size()) - 1;
    }
    void add_item(int menu, cardinal::string item) {
        if (menu >= 0 && menu < static_cast<int>(menus_.size()))
            menus_[static_cast<usize>(menu)].items.push_back(cardinal::move(item));
    }
    void set_on_item(cardinal::function<void(int, int)> f) { on_item_ = cardinal::move(f); }
    int  open_menu() const noexcept { return open_; }

    Vec2 measure(const Constraints& c) override;
    void arrange(const Rect& r) override;      // bar height + open popup
    void paint(PaintContext& ctx) override;
    bool interactive() const noexcept override { return true; }
    bool focusable()   const noexcept override { return true; }
    void on_drag(Vec2 mouse) override { press_ = mouse; press_valid_ = true; }
    void on_click() override;
    void on_focus_changed(bool focused) override { if (!focused) open_ = -1; }

private:
    struct Menu {
        cardinal::string                   title;
        cardinal::vector<cardinal::string> items;
    };
    float title_w(int i) const noexcept;   // span width of title i (incl. padding)
    float title_x(int i) const noexcept;   // span left edge, relative to rect_.left()
    Rect  popup_rect() const noexcept;     // valid while open_ >= 0

    cardinal::vector<Menu> menus_;
    int   open_{-1};
    float bar_h_{24.0f};
    float row_h_{22.0f};
    float font_size_{14.0f};
    float title_pad_{10.0f};
    float item_pad_{10.0f};
    Vec2  press_{};
    bool  press_valid_{false};
    cardinal::function<void(int, int)> on_item_;
};

// Tabs — a header strip of tab labels over a content region showing one page
// at a time. add_tab(label, content) appends a page; non-selected pages are
// hidden (visible=false hides them from paint AND hit-testing). No popups —
// plain in-flow widget.
class Tabs : public Widget {
public:
    Tabs() = default;
    Widget* add_tab(cardinal::string label, cardinal::unique_ptr<Widget> content);
    int  count() const noexcept { return static_cast<int>(labels_.size()); }
    int  selected() const noexcept { return selected_; }
    void set_selected(int i);                  // swaps page visibility + arranges
    void set_on_change(cardinal::function<void(int)> f) { on_change_ = cardinal::move(f); }

    Vec2 measure(const Constraints& c) override;
    void arrange(const Rect& r) override;
    void paint(PaintContext& ctx) override;
    bool interactive() const noexcept override { return true; }
    void on_drag(Vec2 mouse) override { press_ = mouse; press_valid_ = true; }
    void on_click() override;                  // header strip -> select tab

private:
    float tab_w(int i) const noexcept;
    Rect  content_rect() const noexcept {
        return { { rect_.left(), rect_.top() + tab_h_ },
                 { rect_.width(), rect_.height() - tab_h_ } };
    }
    cardinal::vector<cardinal::string> labels_;
    int   selected_{0};
    float tab_h_{26.0f};
    float font_size_{14.0f};
    float tab_pad_{10.0f};
    Vec2  press_{};
    bool  press_valid_{false};
    cardinal::function<void(int)> on_change_;
};

}  // namespace cardinal::cui
