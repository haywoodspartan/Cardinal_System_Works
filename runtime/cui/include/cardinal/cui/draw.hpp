#pragma once

// =============================================================================
// Cardinal UI — graphics-agnostic retained draw list.
//
// A widget's paint() appends DrawCmds; a backend (RHI renderer, ImGui bridge,
// or a headless test recorder) walks cmds() and rasterizes them. Keeping the
// draw list backend-free is what lets the same widget tree render under Vulkan,
// D3D12, an ImGui draw list, or no GPU at all (tests).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/utility.hpp>

#include "geometry.hpp"

namespace cardinal::cui {

enum class DrawKind : u8 { RectFilled, RectStroke, Line, Text };

struct DrawCmd {
    DrawKind         kind{DrawKind::RectFilled};
    Rect             rect{};            // Rect* bounds; Text uses rect.pos as origin
    Vec2             p1{};              // Line end point (rect.pos is the start)
    Color            color{};
    float            thickness{1.0f};   // RectStroke / Line
    float            rounding{0.0f};    // RectFilled / RectStroke corner radius
    float            font_size{0.0f};   // Text
    cardinal::string text;              // Text
    Rect             clip{ {0.0f, 0.0f}, {1.0e9f, 1.0e9f} };   // scissor; default = unclipped
};

class DrawList {
public:
    void clear() noexcept { cmds_.clear(); clip_stack_.clear(); }
    const cardinal::vector<DrawCmd>& cmds() const noexcept { return cmds_; }
    cardinal::usize size() const noexcept { return cmds_.size(); }

    // Clip stack — pushed rects intersect with the current clip, so a scroll
    // container clips its subtree (and nested clips compose). Commands appended
    // while a clip is active carry it; backends apply it as a scissor.
    void push_clip(const Rect& r) {
        clip_stack_.push_back(clip_stack_.empty() ? r : rect_intersect(clip_stack_.back(), r));
    }
    void pop_clip() { if (!clip_stack_.empty()) clip_stack_.pop_back(); }

    void rect_filled(const Rect& r, Color c, float rounding = 0.0f) {
        DrawCmd d;
        d.kind = DrawKind::RectFilled; d.rect = r; d.color = c; d.rounding = rounding;
        push(cardinal::move(d));
    }
    void rect_stroke(const Rect& r, Color c, float thickness = 1.0f, float rounding = 0.0f) {
        DrawCmd d;
        d.kind = DrawKind::RectStroke; d.rect = r; d.color = c;
        d.thickness = thickness; d.rounding = rounding;
        push(cardinal::move(d));
    }
    void line(Vec2 a, Vec2 b, Color c, float thickness = 1.0f) {
        DrawCmd d;
        d.kind = DrawKind::Line; d.rect.pos = a; d.p1 = b; d.color = c; d.thickness = thickness;
        push(cardinal::move(d));
    }
    void text(Vec2 pos, const cardinal::string& s, Color c, float font_size) {
        DrawCmd d;
        d.kind = DrawKind::Text; d.rect.pos = pos; d.text = s; d.color = c; d.font_size = font_size;
        push(cardinal::move(d));
    }

    // Count of a given command kind (handy for tests / diagnostics).
    cardinal::usize count(DrawKind k) const noexcept {
        cardinal::usize n = 0;
        for (const auto& c : cmds_) if (c.kind == k) ++n;
        return n;
    }

private:
    Rect cur_clip() const noexcept {
        return clip_stack_.empty() ? Rect{ {0.0f, 0.0f}, {1.0e9f, 1.0e9f} } : clip_stack_.back();
    }
    void push(DrawCmd d) {
        d.clip = cur_clip();
        cmds_.push_back(cardinal::move(d));
    }
    cardinal::vector<DrawCmd> cmds_;
    cardinal::vector<Rect>    clip_stack_;
};

}  // namespace cardinal::cui
