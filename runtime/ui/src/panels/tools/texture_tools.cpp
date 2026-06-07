// =============================================================================
// Studio — Texture tools panel implementation.
// =============================================================================
#include "texture_tools.hpp"

#include <cardinal/edit/tex_ops.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/std/algorithm.hpp>
#include <cardinal/core/std/cstdio.hpp>

namespace cardinal::ui::panels::texture_tools_panel {

namespace {

cardinal::edit::tex_ops::Color rgba_color(const float c[4]) {
    auto u = [](float f) {
        return static_cast<cardinal::u8>(cardinal::clamp(f, 0.0f, 1.0f) * 255.0f);
    };
    return { u(c[0]), u(c[1]), u(c[2]), u(c[3]) };
}

void regenerate(State& s) {
    namespace t = cardinal::edit::tex_ops;
    if (s.size < 4) s.size = 4;
    if (s.size > 1024) s.size = 1024;
    switch (s.gen) {
        case GenKind::Solid:
            s.rgba_cache = t::solid(s.size, s.size, rgba_color(s.solid_rgba));
            break;
        case GenKind::Checker:
            s.rgba_cache = t::checker(s.size, s.size,
                s.checker_cells_x, s.checker_cells_y,
                rgba_color(s.solid_rgba), rgba_color(s.grad_b_rgba));
            break;
        case GenKind::GradientLinear:
            s.rgba_cache = t::gradient_linear(s.size, s.size,
                rgba_color(s.grad_a_rgba), rgba_color(s.grad_b_rgba),
                static_cast<t::Axis>(s.grad_axis));
            break;
        case GenKind::NoiseValue:
            s.rgba_cache = t::noise_value(s.size, s.size, s.seed,
                                          s.noise_scale, s.noise_contrast);
            break;
        case GenKind::NoiseFractal:
            s.rgba_cache = t::noise_fractal(s.size, s.size, s.seed,
                                            s.noise_scale, s.fractal_octaves,
                                            s.fractal_persistence);
            break;
        case GenKind::Voronoi:
            s.rgba_cache = t::voronoi_cells(s.size, s.size, s.seed, s.voronoi_sites);
            break;
    }
    if (s.post_grayscale) t::to_grayscale(s.rgba_cache, s.size, s.size);
    if (s.post_invert)    t::invert      (s.rgba_cache, s.size, s.size);
    if (s.post_levels)
        t::levels(s.rgba_cache, s.size, s.size,
                  s.levels_black, s.levels_white, s.levels_gamma);
    s.dirty = false;
}

void preview(const State& s) {
    // Render the cache as colored cells via ImGui's draw list. Cheap up to
    // ~256x256; over that the cell-overdraw cost shows up. The preview
    // box is fixed at min(panel width, 256).
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    float side = cardinal::min(256.0f, avail.x);
    if (side < 64.0f) side = 64.0f;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float cell = side / static_cast<float>(s.size);
    for (cardinal::u32 y = 0; y < s.size; ++y) {
        for (cardinal::u32 x = 0; x < s.size; ++x) {
            const cardinal::usize i = (static_cast<cardinal::usize>(y) * s.size + x) * 4;
            const ImU32 col = IM_COL32(s.rgba_cache[i+0], s.rgba_cache[i+1],
                                       s.rgba_cache[i+2], s.rgba_cache[i+3]);
            const ImVec2 p0(origin.x + x * cell,         origin.y + y * cell);
            const ImVec2 p1(origin.x + (x + 1) * cell,   origin.y + (y + 1) * cell);
            dl->AddRectFilled(p0, p1, col);
        }
    }
    // Border.
    dl->AddRect(origin, ImVec2(origin.x + side, origin.y + side),
                IM_COL32(60, 60, 60, 255));
    ImGui::Dummy(ImVec2(side, side));
}

void export_ppm(const State& s) {
    FILE* f = cardinal::fopen(s.export_path, "wb");
    if (f == nullptr) return;
    cardinal::fprintf(f, "P6\n%u %u\n255\n", s.size, s.size);
    for (cardinal::u32 y = 0; y < s.size; ++y) {
        for (cardinal::u32 x = 0; x < s.size; ++x) {
            const cardinal::usize i = (static_cast<cardinal::usize>(y) * s.size + x) * 4;
            cardinal::fwrite(&s.rgba_cache[i], 1, 3, f);  // RGB only (drop A)
        }
    }
    cardinal::fclose(f);
}

}  // namespace

void draw(State* st, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Texture Tools", p_open,
                      0)) {
        ImGui::End();
        return;
    }
    if (st == nullptr) {
        ImGui::TextDisabled("(no state)");
        ImGui::End();
        return;
    }
    st->export_clicked = false;

    bool changed = st->dirty;
    const char* gens[] = { "Solid", "Checker", "Gradient",
                           "Value noise", "Fractal noise", "Voronoi" };
    int g = static_cast<int>(st->gen);
    if (ImGui::Combo("Generator", &g, gens, IM_ARRAYSIZE(gens))) {
        st->gen = static_cast<GenKind>(g);
        changed = true;
    }
    {
        int sz = static_cast<int>(st->size);
        if (ImGui::SliderInt("Size", &sz, 16, 512)) {
            st->size = static_cast<cardinal::u32>(sz);
            changed = true;
        }
        int sd = static_cast<int>(st->seed);
        if (ImGui::InputInt("Seed", &sd)) {
            st->seed = static_cast<cardinal::u32>(sd < 0 ? 0 : sd);
            changed = true;
        }
    }
    switch (st->gen) {
        case GenKind::Solid:
            if (ImGui::ColorEdit4("Color", st->solid_rgba)) changed = true;
            break;
        case GenKind::Checker: {
            if (ImGui::ColorEdit4("A", st->solid_rgba)) changed = true;
            if (ImGui::ColorEdit4("B", st->grad_b_rgba)) changed = true;
            int cx = static_cast<int>(st->checker_cells_x);
            int cy = static_cast<int>(st->checker_cells_y);
            if (ImGui::SliderInt("Cells X", &cx, 1, 64)) {
                st->checker_cells_x = static_cast<cardinal::u32>(cx); changed = true;
            }
            if (ImGui::SliderInt("Cells Y", &cy, 1, 64)) {
                st->checker_cells_y = static_cast<cardinal::u32>(cy); changed = true;
            }
        } break;
        case GenKind::GradientLinear: {
            if (ImGui::ColorEdit4("A", st->grad_a_rgba)) changed = true;
            if (ImGui::ColorEdit4("B", st->grad_b_rgba)) changed = true;
            const char* axes[] = { "X", "Y", "Diagonal" };
            int a = static_cast<int>(st->grad_axis);
            if (ImGui::Combo("Axis", &a, axes, IM_ARRAYSIZE(axes))) {
                st->grad_axis = static_cast<cardinal::u32>(a); changed = true;
            }
        } break;
        case GenKind::NoiseValue: {
            if (ImGui::SliderFloat("Scale",    &st->noise_scale, 0.5f, 64.0f, "%.2f")) changed = true;
            if (ImGui::SliderFloat("Contrast", &st->noise_contrast, 0.1f, 4.0f, "%.2f")) changed = true;
        } break;
        case GenKind::NoiseFractal: {
            if (ImGui::SliderFloat("Base scale", &st->noise_scale, 0.5f, 32.0f, "%.2f")) changed = true;
            int oct = static_cast<int>(st->fractal_octaves);
            if (ImGui::SliderInt("Octaves", &oct, 1, 8)) {
                st->fractal_octaves = static_cast<cardinal::u32>(oct); changed = true;
            }
            if (ImGui::SliderFloat("Persistence", &st->fractal_persistence, 0.1f, 0.95f, "%.2f")) changed = true;
        } break;
        case GenKind::Voronoi: {
            int sites = static_cast<int>(st->voronoi_sites);
            if (ImGui::SliderInt("Sites", &sites, 4, 256)) {
                st->voronoi_sites = static_cast<cardinal::u32>(sites); changed = true;
            }
        } break;
    }

    ImGui::Separator();
    ImGui::TextDisabled("Post ops");
    if (ImGui::Checkbox("Grayscale", &st->post_grayscale)) changed = true;
    ImGui::SameLine();
    if (ImGui::Checkbox("Invert", &st->post_invert)) changed = true;
    if (ImGui::Checkbox("Levels", &st->post_levels)) changed = true;
    if (st->post_levels) {
        if (ImGui::SliderFloat("Black", &st->levels_black, 0.0f, 1.0f, "%.2f")) changed = true;
        if (ImGui::SliderFloat("White", &st->levels_white, 0.0f, 1.0f, "%.2f")) changed = true;
        if (ImGui::SliderFloat("Gamma", &st->levels_gamma, 0.1f, 4.0f, "%.2f")) changed = true;
    }

    if (changed) regenerate(*st);

    ImGui::Separator();
    preview(*st);

    ImGui::Separator();
    ImGui::InputText("Path##exp", st->export_path, sizeof(st->export_path));
    if (ImGui::Button("Export PPM")) {
        export_ppm(*st);
        st->export_clicked = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Writes a portable pixmap (PPM) — view with most\n"
                          "image tools (XnView, IrfanView, GIMP, etc.).");
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::texture_tools_panel
