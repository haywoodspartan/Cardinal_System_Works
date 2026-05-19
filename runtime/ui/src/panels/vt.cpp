// =============================================================================
// Studio — Virtual Texture residency panel implementation.
// =============================================================================
#include "vt.hpp"

#include <cardinal/vt/vt.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cstdio.hpp>

namespace cardinal::ui::panels::vt_panel {

namespace {

ImU32 status_color(cardinal::vt::TileStatus s) {
    using S = cardinal::vt::TileStatus;
    switch (s) {
        case S::NotResident: return IM_COL32(40,  40,  40,  255);
        case S::Pending:     return IM_COL32(180, 160,  60, 255);
        case S::Resident:    return IM_COL32( 80, 200, 100, 255);
        case S::Failed:      return IM_COL32(220,  80,  80, 255);
    }
    return IM_COL32_WHITE;
}

void draw_heatmap(const cardinal::vt::PageTable& pt, cardinal::u32 mip, float side_px) {
    const cardinal::u32 w = pt.width_tiles(mip);
    const cardinal::u32 h = pt.height_tiles(mip);
    if (w == 0 || h == 0) {
        ImGui::TextDisabled("(empty)");
        return;
    }
    const float cell = side_px / static_cast<float>(cardinal::max(w, h));
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (cardinal::u32 y = 0; y < h; ++y) {
        for (cardinal::u32 x = 0; x < w; ++x) {
            const auto lk = pt.lookup(mip, y, x);
            const ImVec2 p0(origin.x + x * cell, origin.y + y * cell);
            const ImVec2 p1(origin.x + (x + 1) * cell, origin.y + (y + 1) * cell);
            dl->AddRectFilled(p0, p1, status_color(lk.status));
        }
    }
    dl->AddRect(origin, ImVec2(origin.x + w * cell, origin.y + h * cell),
                IM_COL32(80, 80, 80, 255));
    ImGui::Dummy(ImVec2(w * cell, h * cell));
}

}  // namespace

void draw(cardinal::vt::System* sys, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Virtual Textures", p_open,
                      ImGuiWindowFlags_NoMove)) {
        ImGui::End();
        return;
    }
    if (sys == nullptr) {
        ImGui::TextDisabled("(no vt::System bound)");
        ImGui::End();
        return;
    }

    const auto stats = sys->stats();
    const auto& cache = stats.cache;
    const cardinal::u64 lookups = cache.hits + cache.misses;
    const double hit_ratio = lookups > 0
        ? 100.0 * static_cast<double>(cache.hits) / static_cast<double>(lookups)
        : 0.0;
    const double dedup_ratio = cache.inserts > 0
        ? 100.0 * static_cast<double>(cache.dedup_hits) / static_cast<double>(cache.inserts)
        : 0.0;

    if (ImGui::CollapsingHeader("Pool", ImGuiTreeNodeFlags_DefaultOpen)) {
        const double bytes_per_tile = static_cast<double>(cardinal::vt::kTileBytesRGBA);
        const double pool_mib       = bytes_per_tile * cache.capacity / (1024.0 * 1024.0);
        const double resident_mib   = bytes_per_tile * cache.resident / (1024.0 * 1024.0);
        ImGui::Text("Capacity:  %u slots (%.1f MiB)",  cache.capacity,  pool_mib);
        ImGui::Text("Resident:  %u slots (%.1f MiB)",  cache.resident,  resident_mib);
        ImGui::Text("Free:      %u slots",              cache.free_slots);
        const float frac = cache.capacity > 0
            ? static_cast<float>(cache.resident) / static_cast<float>(cache.capacity)
            : 0.0f;
        char ov[64];
        cardinal::snprintf(ov, sizeof(ov), "%u / %u (%.1f%%)",
            cache.resident, cache.capacity, frac * 100.0f);
        ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0), ov);
    }

    if (ImGui::CollapsingHeader("Cache counters", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Hits:        %llu",        static_cast<unsigned long long>(cache.hits));
        ImGui::Text("Misses:      %llu",        static_cast<unsigned long long>(cache.misses));
        ImGui::Text("Hit ratio:   %.1f%%",      hit_ratio);
        ImGui::Text("Inserts:     %llu",        static_cast<unsigned long long>(cache.inserts));
        ImGui::Text("Dedup hits:  %llu (%.1f%%)",
            static_cast<unsigned long long>(cache.dedup_hits), dedup_ratio);
        ImGui::Text("Evictions:   %llu",        static_cast<unsigned long long>(cache.evictions));
    }

    if (ImGui::CollapsingHeader("Request queue", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Seen:        %llu",  static_cast<unsigned long long>(stats.requests_seen));
        ImGui::Text("Processed:   %llu",  static_cast<unsigned long long>(stats.requests_processed));
        ImGui::Text("Dropped:     %llu",  static_cast<unsigned long long>(stats.requests_dropped));
        ImGui::Text("Prefetched:  %llu",  static_cast<unsigned long long>(stats.prefetch_added));
        const cardinal::u64 backlog = stats.requests_seen > stats.requests_processed
            ? stats.requests_seen - stats.requests_processed : 0;
        ImGui::Text("Backlog:     %llu",  static_cast<unsigned long long>(backlog));
    }

    if (ImGui::CollapsingHeader("Per-VT", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Registered: %u VTs", stats.vt_count);
        // Walk vts by id 0..N (slot count grows by id).
        for (cardinal::u32 id = 0; id < 64; ++id) {
            auto vt = sys->get_vt(static_cast<cardinal::vt::VtId>(id));
            if (vt == nullptr) continue;
            const auto& pt = vt->page_table();
            ImGui::PushID(static_cast<int>(id));
            char hdr[80];
            cardinal::snprintf(hdr, sizeof(hdr),
                "vt %u — %u x %u tiles, %u mips, resident %u, pending %u",
                id, vt->desc().width_tiles, vt->desc().height_tiles,
                pt.mip_count(), pt.resident_count(), pt.pending_count());
            if (ImGui::TreeNode(hdr)) {
                if (ImGui::BeginTable("##vt_mips", 4,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                        ImGuiTableFlags_SizingStretchProp))
                {
                    ImGui::TableSetupColumn("mip",   ImGuiTableColumnFlags_WidthFixed, 50.0f);
                    ImGui::TableSetupColumn("tiles", ImGuiTableColumnFlags_WidthFixed, 110.0f);
                    ImGui::TableSetupColumn("MiB",   ImGuiTableColumnFlags_WidthFixed, 80.0f);
                    ImGui::TableSetupColumn("notes", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableHeadersRow();
                    for (cardinal::u32 m = 0; m < pt.mip_count(); ++m) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0); ImGui::Text("%u", m);
                        ImGui::TableSetColumnIndex(1); ImGui::Text("%u x %u",
                            pt.width_tiles(m), pt.height_tiles(m));
                        ImGui::TableSetColumnIndex(2);
                        const double mib =
                            static_cast<double>(cardinal::vt::kTileBytesRGBA) *
                            pt.width_tiles(m) * pt.height_tiles(m) / (1024.0 * 1024.0);
                        ImGui::Text("%.1f", mib);
                        ImGui::TableSetColumnIndex(3); ImGui::TextDisabled("(virtual)");
                    }
                    ImGui::EndTable();
                }
                ImGui::TextDisabled("Mip-0 residency map:");
                draw_heatmap(pt, 0, cardinal::min(256.0f,
                    ImGui::GetContentRegionAvail().x));
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::vt_panel
