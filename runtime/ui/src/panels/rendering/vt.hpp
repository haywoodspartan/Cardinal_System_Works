#pragma once

// =============================================================================
// Studio — Virtual Texture residency panel.
//
// Reads a cardinal::vt::System for live cache + per-VT stats and renders:
//   - Pool capacity / residency / free
//   - Hit/miss/dedup/evict counters + ratios
//   - Request-queue throughput (seen / processed / dropped / prefetched)
//   - Per-VT residency by mip
//   - Optional residency heat-map (mip 0 only): paints each tile cell
//     by its TileStatus
// =============================================================================

namespace cardinal::vt { class System; }

namespace cardinal::ui::panels::vt_panel {

void draw(cardinal::vt::System* system,
          const char* title = "Virtual Textures",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::vt_panel
