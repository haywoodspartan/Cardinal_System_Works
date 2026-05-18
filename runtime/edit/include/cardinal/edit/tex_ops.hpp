#pragma once

// =============================================================================
// Cardinal — CPU procedural texture generators + image ops.
//
// Output buffers are RGBA8 (4 bytes per pixel, row-major). Width × height
// drives the size; functions allocate the result vector.
//
// Generators:
//   solid                — flat color
//   checker              — N×N alternating cells
//   gradient_linear      — A→B along an axis
//   noise_value          — deterministic value noise (hash-based)
//   noise_fractal        — N octaves of value noise
//   voronoi_cells        — closest-feature distance
//
// Operations:
//   to_grayscale         — luminance copy
//   invert               — 255 - r/g/b per channel (alpha preserved)
//   levels               — black/white/gamma point adjustment
//   channel_swap         — reorder rgba
//   compose              — alpha-over a foreground onto a background
//
// Math is pragmatic — these are editor-time helpers, not a Substance clone.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <vector>

namespace cardinal::edit::tex_ops {

struct Color { u8 r, g, b, a; };

// -------- Generators -------------------------------------------------------

std::vector<u8> solid(u32 w, u32 h, Color c);

std::vector<u8> checker(u32 w, u32 h, u32 cells_x, u32 cells_y,
                        Color a, Color b);

enum class Axis { X, Y, Diagonal };
std::vector<u8> gradient_linear(u32 w, u32 h, Color a, Color b, Axis axis);

std::vector<u8> noise_value (u32 w, u32 h, u32 seed = 1337,
                             float scale = 8.0f, float contrast = 1.0f);

std::vector<u8> noise_fractal(u32 w, u32 h, u32 seed = 1337,
                              float base_scale = 4.0f,
                              u32   octaves = 4,
                              float persistence = 0.5f);

std::vector<u8> voronoi_cells(u32 w, u32 h, u32 seed = 1337, u32 site_count = 32);

// -------- Operations -------------------------------------------------------

void to_grayscale(std::vector<u8>& rgba, u32 w, u32 h);
void invert      (std::vector<u8>& rgba, u32 w, u32 h);

// black_pt / white_pt in [0..1]; inputs <= black_pt map to 0, >= white_pt
// map to 255, with `gamma` applied to the rescaled middle range.
void levels(std::vector<u8>& rgba, u32 w, u32 h,
            float black_pt = 0.0f, float white_pt = 1.0f, float gamma = 1.0f);

// channel_swap("rgba") = identity; "bgra" = swap R/B; "rgbr" = put R into A; etc.
// Each character must be one of r/g/b/a. Strings shorter than 4 chars are an error.
void channel_swap(std::vector<u8>& rgba, u32 w, u32 h, const char* layout);

// dst = compose(dst, src) using src.a as alpha.
void compose_alpha(std::vector<u8>& dst, const std::vector<u8>& src,
                   u32 w, u32 h);

// Single-pixel access helpers — no bounds checking.
inline Color get_px(const std::vector<u8>& img, u32 w, u32 x, u32 y) noexcept {
    const usize i = (static_cast<usize>(y) * w + x) * 4;
    return Color{ img[i+0], img[i+1], img[i+2], img[i+3] };
}
inline void put_px(std::vector<u8>& img, u32 w, u32 x, u32 y, Color c) noexcept {
    const usize i = (static_cast<usize>(y) * w + x) * 4;
    img[i+0] = c.r; img[i+1] = c.g; img[i+2] = c.b; img[i+3] = c.a;
}

}  // namespace cardinal::edit::tex_ops
