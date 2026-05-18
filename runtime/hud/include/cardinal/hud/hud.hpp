#pragma once

// =============================================================================
// Cardinal — In-game HUD primitives.
//
// Separate from the Studio editor UI. Where Studio uses ImGui windows + dock,
// the HUD draws simple retained-mode 2D primitives directly on top of the
// game viewport: bars (health/stamina), text labels, crosshairs, minimaps,
// damage flashes, etc.
//
// The HUD operates on a backing draw list the host owns (typically
// `ImGui::GetForegroundDrawList()` so HUD strokes always sit above
// everything else). We don't pull in imgui in the public header — instead
// we use a void* pointer + a thin wrapper.
//
// Coordinates are screen pixels with origin top-left.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>

namespace cardinal::hud {

struct Color { u8 r, g, b, a; };
inline constexpr Color rgb (u8 r, u8 g, u8 b)         noexcept { return {r,g,b,255}; }
inline constexpr Color rgba(u8 r, u8 g, u8 b, u8 a)   noexcept { return {r,g,b,a}; }

// ---------------------------------------------------------------------------
// HUD — owns a per-frame draw queue. The host calls begin_frame() with the
// screen dimensions, queues primitives, then render() flushes them onto
// the supplied draw list (an ImGui::GetForegroundDrawList()-style pointer).
//
// We keep the draw list pointer as void* so this header doesn't pull in
// imgui — the cpp casts to ImDrawList*.
// ---------------------------------------------------------------------------
class Hud {
public:
    Hud();

    void begin_frame(u32 screen_w, u32 screen_h) noexcept;

    // ---- Text ------------------------------------------------------
    // Optional `wrap_w` 0 = no wrap.
    void text(float x, float y, const cardinal::string& s, Color c = {255,255,255,255}) noexcept;

    // ---- Bars (HP / mana / stamina / progress) --------------------
    // Filled fraction 0..1. Optional border + label.
    void bar(float x, float y, float w, float h,
             float fill_fraction,
             Color fg = {0, 200, 60, 255},
             Color bg = {30, 30, 30, 200}) noexcept;
    void bar_with_label(float x, float y, float w, float h,
                        float fill_fraction,
                        const cardinal::string& label,
                        Color fg = {0, 200, 60, 255}) noexcept;

    // ---- Reticles / crosshairs -------------------------------------
    // Centered at (cx, cy). `inner_radius` < `outer_radius`; classic
    // 4-tick crosshair (T/B/L/R) — call multiple times for compound
    // shapes (kill confirms, hit markers).
    void crosshair(float cx, float cy,
                   float inner_radius, float outer_radius,
                   Color c = {255, 255, 255, 220}, float thickness = 1.5f) noexcept;
    void hit_marker(float cx, float cy, float size = 12.0f,
                    Color c = {255, 80, 80, 230}) noexcept;

    // ---- Damage / screen flashes ----------------------------------
    // Tints the entire screen with a vignette mask. `intensity` 0..1.
    void damage_flash(float intensity = 0.5f,
                      Color c = {255, 30, 30, 180}) noexcept;

    // ---- Minimap ---------------------------------------------------
    // Renders a square map in the corner. World-space coords are mapped
    // into screen via `world_origin` (top-left world coord) + `world_size`.
    // Each marker = (world x, world y, color).
    struct MapMarker { float wx, wy; Color color; };
    void minimap(float x, float y, float side_px,
                 float world_origin_x, float world_origin_y,
                 float world_size,
                 const cardinal::vector<MapMarker>& markers,
                 Color border = {200, 200, 200, 220},
                 Color bg     = { 10,  10,  20, 180}) noexcept;

    // ---- Toast notifications --------------------------------------
    // Stacks small fade-out messages at top-right (or wherever you call).
    // Add via push_toast(); the HUD ticks them out over `duration_s`.
    struct ToastOpts {
        float duration_s{2.5f};
        Color color{255, 255, 255, 255};
    };
    void push_toast(const cardinal::string& message, ToastOpts opts = {});
    void tick_toasts(float dt) noexcept;
    void render_toasts(float anchor_x, float anchor_y) noexcept;   // top-right of group

    // ---- Render to backing draw list ------------------------------
    // `imgui_draw_list` should be (void*)ImGui::GetForegroundDrawList()
    // (or any ImDrawList* — but the foreground list always renders on
    // top of every window).
    void render(void* imgui_draw_list) noexcept;

private:
    struct Toast {
        cardinal::string message;
        float       remaining{0.0f};
        float       duration {0.0f};
        Color       color    {255, 255, 255, 255};
    };
    // Recorded primitive (small union-y struct).
    enum class PrimKind : u32 {
        Text, RectFilled, Rect, Line, Circle, CircleFilled,
        Triangle, Vignette
    };
    struct Prim {
        PrimKind kind;
        float    a{}, b{}, c{}, d{}, e{}, f{};
        float    g{}, h{};
        Color    col{};
        float    thickness{1.0f};
        cardinal::string text;
    };

    u32                screen_w_{0};
    u32                screen_h_{0};
    cardinal::vector<Prim>  prims_;
    cardinal::vector<Toast> toasts_;
};

}  // namespace cardinal::hud
