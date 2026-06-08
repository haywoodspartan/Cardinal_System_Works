#pragma once

// =============================================================================
// Cardinal UI — theme / style tokens.
//
// Colors + a few metrics that widgets read at paint time. Layout sizes are
// carried by widgets themselves (so the measure/arrange passes stay theme-
// independent); the theme only drives appearance.
// =============================================================================

#include "geometry.hpp"

namespace cardinal::cui {

struct Theme {
    // Surfaces.
    Color window_bg     { 28,  28,  32, 255 };
    Color panel_bg      { 40,  40,  46, 255 };
    Color border        { 64,  64,  72, 255 };
    // Text.
    Color text          { 224, 224, 230, 255 };
    Color text_dim      { 150, 150, 158, 255 };
    // Controls (buttons / checkboxes / slider tracks).
    Color control       { 58,  58,  66, 255 };
    Color control_hot   { 78,  78,  88, 255 };
    Color control_active{ 96, 110, 150, 255 };
    Color accent        { 90, 140, 230, 255 };

    // Metrics.
    float padding  { 6.0f };
    float spacing  { 4.0f };
    float rounding { 3.0f };
    float font_size{ 14.0f };
};

// Process-wide default (dark) theme.
inline const Theme& default_theme() noexcept {
    static const Theme t{};
    return t;
}

}  // namespace cardinal::cui
