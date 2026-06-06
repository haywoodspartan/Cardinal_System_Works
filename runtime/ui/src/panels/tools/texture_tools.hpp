#pragma once

// =============================================================================
// Studio — Texture tools panel.
//
// Generates a procedural texture into the panel's preview buffer using the
// cardinal::edit::tex_ops generators. The user can pick the generator type
// (solid / checker / gradient / value noise / fractal noise / voronoi),
// tweak parameters, and apply post-ops (grayscale / invert / levels /
// channel swap).
//
// The preview is shown as an ImGui::Image. The panel does NOT own any GPU
// texture handle directly — it asks the host to upload its CPU buffer
// when needed. For now, the panel just renders the buffer to a software
// preview via ImGui's draw list API (fast enough for ≤ 256x256 previews).
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>

namespace cardinal::ui::panels::texture_tools_panel {

enum class GenKind : cardinal::u32 {
    Solid, Checker, GradientLinear, NoiseValue, NoiseFractal, Voronoi
};

struct State {
    GenKind     gen{GenKind::Checker};
    cardinal::u32 size{128};                    // preview is size x size
    cardinal::u32 seed{1337};

    float       solid_rgba[4]{1,1,1,1};
    cardinal::u32 checker_cells_x{4};
    cardinal::u32 checker_cells_y{4};
    float       grad_a_rgba[4]{0,0,0,1};
    float       grad_b_rgba[4]{1,1,1,1};
    cardinal::u32 grad_axis{0};                 // 0=X, 1=Y, 2=Diag
    float       noise_scale{8.0f};
    float       noise_contrast{1.0f};
    cardinal::u32 fractal_octaves{4};
    float       fractal_persistence{0.5f};
    cardinal::u32 voronoi_sites{32};

    // Post ops -- applied to the generated buffer in order if true.
    bool        post_grayscale{false};
    bool        post_invert{false};
    float       levels_black{0.0f};
    float       levels_white{1.0f};
    float       levels_gamma{1.0f};
    bool        post_levels{false};

    // Live preview cache: regenerated whenever the user changes a knob.
    cardinal::vector<cardinal::u8> rgba_cache;
    bool                       dirty{true};
    bool                       export_clicked{false};
    char                       export_path[512]{"texture.ppm"};
};

void draw(State* state, const char* title = "Texture Tools",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::texture_tools_panel
