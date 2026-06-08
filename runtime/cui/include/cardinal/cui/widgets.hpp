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
// More widgets (text input, list/table, dock, tabs) come in later phases.
// =============================================================================

#include <cardinal/core/types.hpp>

#include "widget.hpp"

namespace cardinal::cui {

// Linear container.
class Stack : public Widget {
public:
    explicit Stack(Axis axis, float spacing = 4.0f, Align cross = Align::Stretch)
        : axis_(axis), spacing_(spacing), cross_(cross) {}
    Vec2 measure(const Constraints& c) override;
    void arrange(const Rect& r) override;
    void paint(PaintContext& ctx) override { paint_children(ctx); }
private:
    Axis  axis_;
    float spacing_;
    Align cross_;
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

}  // namespace cardinal::cui
