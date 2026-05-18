#include <cardinal/hud/hud.hpp>

#include <imgui.h>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::hud {

namespace {

inline ImU32 to_imu32(Color c) noexcept {
    return IM_COL32(c.r, c.g, c.b, c.a);
}

}  // namespace

Hud::Hud() {
    prims_.reserve(64);
    toasts_.reserve(8);
}

void Hud::begin_frame(u32 sw, u32 sh) noexcept {
    screen_w_ = sw; screen_h_ = sh;
    prims_.clear();
}

void Hud::text(float x, float y, const cardinal::string& s, Color c) noexcept {
    Prim p{};
    p.kind = PrimKind::Text;
    p.a = x; p.b = y;
    p.col = c;
    p.text = s;
    prims_.push_back(cardinal::move(p));
}

void Hud::bar(float x, float y, float w, float h,
              float fill_fraction,
              Color fg, Color bg) noexcept
{
    if (fill_fraction < 0.0f) fill_fraction = 0.0f;
    if (fill_fraction > 1.0f) fill_fraction = 1.0f;
    // Background.
    Prim bg_p{};
    bg_p.kind = PrimKind::RectFilled;
    bg_p.a = x;     bg_p.b = y;
    bg_p.c = x + w; bg_p.d = y + h;
    bg_p.col = bg;
    prims_.push_back(cardinal::move(bg_p));
    // Foreground.
    if (fill_fraction > 0.0f) {
        Prim fg_p{};
        fg_p.kind = PrimKind::RectFilled;
        fg_p.a = x;                    fg_p.b = y;
        fg_p.c = x + w * fill_fraction; fg_p.d = y + h;
        fg_p.col = fg;
        prims_.push_back(cardinal::move(fg_p));
    }
    // Border.
    Prim border{};
    border.kind = PrimKind::Rect;
    border.a = x;     border.b = y;
    border.c = x + w; border.d = y + h;
    border.col = {255, 255, 255, 100};
    border.thickness = 1.0f;
    prims_.push_back(cardinal::move(border));
}

void Hud::bar_with_label(float x, float y, float w, float h,
                          float fill_fraction,
                          const cardinal::string& label,
                          Color fg) noexcept
{
    bar(x, y, w, h, fill_fraction, fg);
    text(x + 4.0f, y + h * 0.5f - 7.0f, label, {255, 255, 255, 220});
}

void Hud::crosshair(float cx, float cy,
                    float inner_radius, float outer_radius,
                    Color c, float thickness) noexcept
{
    auto line = [&](float x0, float y0, float x1, float y1) {
        Prim p{};
        p.kind = PrimKind::Line;
        p.a = x0; p.b = y0; p.c = x1; p.d = y1;
        p.col = c; p.thickness = thickness;
        prims_.push_back(cardinal::move(p));
    };
    line(cx - outer_radius, cy, cx - inner_radius, cy);
    line(cx + inner_radius, cy, cx + outer_radius, cy);
    line(cx, cy - outer_radius, cx, cy - inner_radius);
    line(cx, cy + inner_radius, cx, cy + outer_radius);
}

void Hud::hit_marker(float cx, float cy, float size, Color c) noexcept
{
    auto line = [&](float x0, float y0, float x1, float y1) {
        Prim p{};
        p.kind = PrimKind::Line;
        p.a = x0; p.b = y0; p.c = x1; p.d = y1;
        p.col = c; p.thickness = 2.5f;
        prims_.push_back(cardinal::move(p));
    };
    line(cx - size, cy - size, cx - size * 0.4f, cy - size * 0.4f);
    line(cx + size, cy - size, cx + size * 0.4f, cy - size * 0.4f);
    line(cx - size, cy + size, cx - size * 0.4f, cy + size * 0.4f);
    line(cx + size, cy + size, cx + size * 0.4f, cy + size * 0.4f);
}

void Hud::damage_flash(float intensity, Color c) noexcept {
    if (intensity <= 0.0f) return;
    Prim p{};
    p.kind = PrimKind::Vignette;
    p.col  = c;
    p.col.a = static_cast<u8>(static_cast<float>(c.a) * intensity);
    prims_.push_back(cardinal::move(p));
}

void Hud::minimap(float x, float y, float side_px,
                  float wox, float woy, float ws,
                  const cardinal::vector<MapMarker>& markers,
                  Color border, Color bg) noexcept
{
    Prim bgp{};
    bgp.kind = PrimKind::RectFilled;
    bgp.a = x;            bgp.b = y;
    bgp.c = x + side_px;  bgp.d = y + side_px;
    bgp.col = bg;
    prims_.push_back(cardinal::move(bgp));

    for (const auto& m : markers) {
        const float t_x = (m.wx - wox) / ws;
        const float t_y = (m.wy - woy) / ws;
        if (t_x < 0.0f || t_x > 1.0f || t_y < 0.0f || t_y > 1.0f) continue;
        Prim dot{};
        dot.kind = PrimKind::CircleFilled;
        dot.a = x + t_x * side_px;
        dot.b = y + t_y * side_px;
        dot.c = 3.5f;       // radius
        dot.col = m.color;
        prims_.push_back(cardinal::move(dot));
    }

    Prim bd{};
    bd.kind = PrimKind::Rect;
    bd.a = x;            bd.b = y;
    bd.c = x + side_px;  bd.d = y + side_px;
    bd.col = border; bd.thickness = 1.5f;
    prims_.push_back(cardinal::move(bd));
}

void Hud::push_toast(const cardinal::string& message, ToastOpts opts) {
    Toast t{};
    t.message  = message;
    t.duration = opts.duration_s;
    t.remaining = opts.duration_s;
    t.color    = opts.color;
    toasts_.push_back(cardinal::move(t));
}

void Hud::tick_toasts(float dt) noexcept {
    for (auto& t : toasts_) t.remaining -= dt;
    toasts_.erase(cardinal::remove_if(toasts_.begin(), toasts_.end(),
        [](const Toast& t){ return t.remaining <= 0.0f; }),
        toasts_.end());
}

void Hud::render_toasts(float anchor_x, float anchor_y) noexcept {
    float y = anchor_y;
    for (const auto& t : toasts_) {
        const float a = cardinal::min(1.0f, t.remaining / cardinal::max(0.001f, t.duration));
        Color c = t.color;
        c.a = static_cast<u8>(static_cast<float>(c.a) * a);
        text(anchor_x, y, t.message, c);
        y += 18.0f;
    }
}

void Hud::render(void* draw_list) noexcept {
    auto* dl = static_cast<ImDrawList*>(draw_list);
    if (dl == nullptr) return;
    for (const auto& p : prims_) {
        switch (p.kind) {
            case PrimKind::Text:
                dl->AddText({p.a, p.b}, to_imu32(p.col), p.text.c_str());
                break;
            case PrimKind::RectFilled:
                dl->AddRectFilled({p.a, p.b}, {p.c, p.d}, to_imu32(p.col));
                break;
            case PrimKind::Rect:
                dl->AddRect({p.a, p.b}, {p.c, p.d}, to_imu32(p.col),
                            0.0f, 0, p.thickness);
                break;
            case PrimKind::Line:
                dl->AddLine({p.a, p.b}, {p.c, p.d}, to_imu32(p.col), p.thickness);
                break;
            case PrimKind::Circle:
                dl->AddCircle({p.a, p.b}, p.c, to_imu32(p.col), 0, p.thickness);
                break;
            case PrimKind::CircleFilled:
                dl->AddCircleFilled({p.a, p.b}, p.c, to_imu32(p.col));
                break;
            case PrimKind::Triangle:
                dl->AddTriangle({p.a, p.b}, {p.c, p.d}, {p.e, p.f},
                                 to_imu32(p.col), p.thickness);
                break;
            case PrimKind::Vignette: {
                // Radial-ish darkening from the edges, faked as a ring of
                // semi-transparent rects.
                const float w = static_cast<float>(screen_w_);
                const float h = static_cast<float>(screen_h_);
                const ImU32 col = to_imu32(p.col);
                dl->AddRectFilled({0,0}, {w, h * 0.10f}, col);
                dl->AddRectFilled({0, h * 0.90f}, {w, h}, col);
                dl->AddRectFilled({0, h * 0.10f}, {w * 0.10f, h * 0.90f}, col);
                dl->AddRectFilled({w * 0.90f, h * 0.10f}, {w, h * 0.90f}, col);
                break;
            }
        }
    }
}

}  // namespace cardinal::hud
