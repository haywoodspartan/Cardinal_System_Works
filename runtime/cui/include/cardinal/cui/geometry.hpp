#pragma once

// =============================================================================
// Cardinal UI ("cui" / Cardinal Slate) — 2D layout geometry.
//
// This is Cardinal's own retained-mode UI framework (a Slate analog): a widget
// tree with a two-pass box layout, input routing, theming, and a graphics-
// agnostic draw list. It is a LEAF library — it depends only on cardinal::core
// and knows nothing about the engine, the RHI, or the Studio, so it can back
// both the editor and shipped-game UI. A renderer backend (RHI / ImGui / a
// headless test recorder) consumes the draw list; the framework itself is
// fully testable on the CPU with no GPU.
//
// Coordinate space: pixels, origin top-left, +x right, +y down.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/math/math.hpp>

namespace cardinal::cui {

// Reuse the engine's canonical 2D vector (single source of truth — no parallel
// Vec2). Used here for both points and sizes (x = width, y = height).
using Vec2 = cardinal::core::Vec2;

// 8-bit RGBA color (UI-owned, matching the HUD convention).
struct Color {
    u8 r{255}, g{255}, b{255}, a{255};
};
inline constexpr Color rgb (u8 r, u8 g, u8 b)       noexcept { return {r, g, b, 255}; }
inline constexpr Color rgba(u8 r, u8 g, u8 b, u8 a) noexcept { return {r, g, b, a}; }

// Axis-aligned rectangle: top-left origin + size.
struct Rect {
    Vec2 pos{};    // top-left corner
    Vec2 size{};   // (width, height)

    constexpr float left()   const noexcept { return pos.x; }
    constexpr float top()    const noexcept { return pos.y; }
    constexpr float right()  const noexcept { return pos.x + size.x; }
    constexpr float bottom() const noexcept { return pos.y + size.y; }
    constexpr float width()  const noexcept { return size.x; }
    constexpr float height() const noexcept { return size.y; }
    constexpr Vec2  center() const noexcept {
        return { pos.x + size.x * 0.5f, pos.y + size.y * 0.5f };
    }
    constexpr bool contains(Vec2 p) const noexcept {
        return p.x >= pos.x && p.x <= pos.x + size.x &&
               p.y >= pos.y && p.y <= pos.y + size.y;
    }
    // Shrink inward by edge insets (clamped non-negative).
    constexpr Rect inset(float l, float t, float r, float b) const noexcept {
        float w = size.x - l - r; if (w < 0.0f) w = 0.0f;
        float h = size.y - t - b; if (h < 0.0f) h = 0.0f;
        return { { pos.x + l, pos.y + t }, { w, h } };
    }
};

// Edge insets (padding / margin): left, top, right, bottom.
struct Edges {
    float l{0}, t{0}, r{0}, b{0};
    static constexpr Edges all(float v)         noexcept { return {v, v, v, v}; }
    static constexpr Edges xy(float x, float y) noexcept { return {x, y, x, y}; }
    constexpr float horiz() const noexcept { return l + r; }
    constexpr float vert()  const noexcept { return t + b; }
};

// Box-layout constraints: the min/max width+height envelope a child must fit.
struct Constraints {
    float min_w{0}, max_w{1e9f};
    float min_h{0}, max_h{1e9f};

    static constexpr Constraints loose(float w, float h) noexcept { return {0, w, 0, h}; }
    static constexpr Constraints tight(float w, float h) noexcept { return {w, w, h, h}; }

    constexpr float clamp_w(float w) const noexcept {
        return w < min_w ? min_w : (w > max_w ? max_w : w);
    }
    constexpr float clamp_h(float h) const noexcept {
        return h < min_h ? min_h : (h > max_h ? max_h : h);
    }
    // Loosen by an inset (for a padded container's child).
    constexpr Constraints deflate(const Edges& e) const noexcept {
        float mw = max_w - e.horiz(); if (mw < 0.0f) mw = 0.0f;
        float mh = max_h - e.vert();  if (mh < 0.0f) mh = 0.0f;
        return { min_w, mw, min_h, mh };
    }
};

enum class Axis  : u8 { Horizontal, Vertical };
enum class Align : u8 { Start, Center, End, Stretch };

// 9-point anchor for absolute placement within a parent rect — Cardinal's take
// on Pearl Abyss's PAUIBase H/V align model (ComputePos), reworked into a clean
// composable form. Values are laid out row-major (row = index/3 = Top/Mid/Bot,
// col = index%3 = Left/Center/Right) so a container can decode them branch-free.
enum class Anchor : u8 {
    TopLeft = 0, TopCenter,    TopRight,
    CenterLeft, Center,        CenterRight,
    BottomLeft, BottomCenter,  BottomRight,
};
inline constexpr int anchor_col(Anchor a) noexcept { return static_cast<int>(a) % 3; }   // 0 L,1 C,2 R
inline constexpr int anchor_row(Anchor a) noexcept { return static_cast<int>(a) / 3; }   // 0 T,1 M,2 B

// -----------------------------------------------------------------------------
// Text metrics. Until a real font/glyph-atlas system lands (phase 2), widgets
// size text with a fixed advance ratio so layout stays deterministic + testable
// on the CPU. A real renderer can override these once fonts exist.
// -----------------------------------------------------------------------------
// Intersection of two rects (clamped non-negative). Used for nested clipping.
inline Rect rect_intersect(const Rect& a, const Rect& b) noexcept {
    const float x0 = a.left()  > b.left()  ? a.left()  : b.left();
    const float y0 = a.top()   > b.top()   ? a.top()   : b.top();
    float       x1 = a.right()  < b.right()  ? a.right()  : b.right();
    float       y1 = a.bottom() < b.bottom() ? a.bottom() : b.bottom();
    if (x1 < x0) x1 = x0;
    if (y1 < y0) y1 = y0;
    return { { x0, y0 }, { x1 - x0, y1 - y0 } };
}

inline float text_advance(const cardinal::string& s, float font_size) noexcept {
    // 0.75 = the 8x8 bitmap font's advance (6 of 8 columns per glyph); the
    // native cui_rhi renderer draws square pixels at this advance, so layout
    // and rendering agree. (The ImGui-bridge preview measures its own font.)
    return static_cast<float>(s.size()) * font_size * 0.75f;
}
inline float line_height(float font_size) noexcept { return font_size + 2.0f; }

}  // namespace cardinal::cui
