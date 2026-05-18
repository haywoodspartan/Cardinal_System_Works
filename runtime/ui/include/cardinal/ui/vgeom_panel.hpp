#pragma once

// =============================================================================
// Cardinal Studio — Virtualized Geometry diagnostic panel.
//
// Surfaces the runtime state of cardinal::vgeom across the scene:
//
//   - Aggregate row: total master triangles (across all attached
//     meshes) vs total drawn this frame, with the savings ratio
//   - Per-mesh table: name, enabled toggle, master/drawn tris,
//     cluster cut size, frustum-culled count, savings %
//   - Per-row "Attach" button for visible meshes that don't yet have
//     a vgeom hierarchy — one click cooks them
//
// The renderer publishes vgeom::FrameStats per Mesh* each frame via
// scene::vgeom_publish_stats; this panel reads them. No selection
// re-runs here, no GPU work — pure inspection.
//
// Free function — no Studio class dependency. Hosts call
// cardinal::ui::draw_vgeom_panel(scene, &visible) inside their UI block.
// =============================================================================

namespace cardinal::scene { class Scene; }

namespace cardinal::ui {

void draw_vgeom_panel(cardinal::scene::Scene* scene, bool* p_open);

}  // namespace cardinal::ui
