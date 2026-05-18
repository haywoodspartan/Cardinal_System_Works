#pragma once

// =============================================================================
// Studio — Mesh tools panel.
//
// Operates on the currently-selected entity's mesh (or, if nothing's
// selected, lets the user generate a new primitive into a fresh entity).
//
// Buttons:
//   Generate primitive : box / sphere / cylinder / cone / torus / disk / plane
//                        with sliders for the relevant dimensions.
//   Apply op           : subdivide / mirror / decimate / smooth /
//                        recompute-normals (per-axis flags).
//
// The host owns the action -- this panel just collects the request and
// hands it back through the State struct (mirroring the world panel
// pattern). The host's main loop checks `apply_*_clicked` flags after
// the panel call.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::ui::panels::mesh_tools_panel {

enum class PrimitiveKind : cardinal::u32 {
    Box, Sphere, Cylinder, Cone, Torus, Disk, Plane
};

struct State {
    // ----- Primitive generator -----
    PrimitiveKind  prim_kind{PrimitiveKind::Box};
    float          prim_size{1.0f};        // box edge / plane edge / generic
    float          prim_radius{1.0f};      // sphere / cylinder / cone / disk
    float          prim_height{2.0f};      // cylinder / cone
    float          prim_minor{0.3f};       // torus minor
    cardinal::u32  prim_segments{24};
    bool           generate_clicked{false};

    // ----- Edits to selected mesh -----
    cardinal::u32  subdivide_levels{1};
    bool           subdivide_clicked{false};

    cardinal::u32  mirror_axis{0};         // 0=X, 1=Y, 2=Z
    bool           mirror_clicked{false};

    float          decimate_cell{0.25f};
    bool           decimate_clicked{false};

    cardinal::u32  smooth_iterations{2};
    float          smooth_lambda{0.5f};
    bool           smooth_clicked{false};

    bool           recompute_normals_smooth{true};
    bool           recompute_normals_clicked{false};
};

void draw(State* state, bool selection_has_mesh,
          const char* title = "Mesh Tools", bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::mesh_tools_panel
