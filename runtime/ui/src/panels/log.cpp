// =============================================================================
// Studio — log panel implementation.
// =============================================================================
#include "log.hpp"

namespace cardinal::ui::panels::log_panel {

using cardinal::u32;

// ----- Store (cardinal::log::Sink) -----------------------------------------

void Store::on_message(cardinal::log::Level lvl, const char* cat, const char* msg) {
    cardinal::lock_guard lk(mutex_);
    Entry& slot = entries_[head_];
    slot.lvl     = lvl;
    slot.category.assign(cat ? cat : "");
    slot.message.assign(msg ? msg : "");
    head_  = (head_ + 1) % kCapacity;
    if (size_ < kCapacity) ++size_;
}

void Store::snapshot(cardinal::vector<Entry>& out) const {
    cardinal::lock_guard lk(mutex_);
    out.clear();
    out.reserve(size_);
    const size_t start = (head_ + kCapacity - size_) % kCapacity;
    for (size_t i = 0; i < size_; ++i) {
        out.push_back(entries_[(start + i) % kCapacity]);
    }
}

void Store::clear() {
    cardinal::lock_guard lk(mutex_);
    head_ = 0; size_ = 0;
}

// ----- draw ----------------------------------------------------------------

namespace {
struct LevelStyle { ImU32 col; const char* tag; };
LevelStyle level_style(cardinal::log::Level l) {
    switch (l) {
        case cardinal::log::Level::Trace: return { IM_COL32(140,140,140,255), "T" };
        case cardinal::log::Level::Info:  return { IM_COL32(220,220,220,255), "I" };
        case cardinal::log::Level::Warn:  return { IM_COL32(255,200, 80,255), "W" };
        case cardinal::log::Level::Error: return { IM_COL32(255,100,100,255), "E" };
    }
    return { IM_COL32(255,255,255,255), "?" };
}
}  // namespace

void draw(const char* title, bool* p_open, State& state) {
    if (!ImGui::Begin(title ? title : "Log", p_open)) { ImGui::End(); return; }

    if (ImGui::Button("Clear")) state.cstore.clear();
    ImGui::SameLine(); ImGui::Checkbox("Auto-scroll", &state.auto_scroll);
    ImGui::SameLine(); ImGui::SetNextItemWidth(180.0f);
    state.filter.Draw("Filter", 180.0f);

    ImGui::SameLine();
    const char* level_labels[] = { "Trace", "Info", "Warn", "Error" };
    int min_lvl = static_cast<int>(state.min_level);
    ImGui::SetNextItemWidth(80.0f);
    if (ImGui::Combo("##min", &min_lvl, level_labels, IM_ARRAYSIZE(level_labels))) {
        state.min_level = static_cast<cardinal::log::Level>(min_lvl);
    }

    ImGui::Separator();

    state.cstore.snapshot(state.scratch_entries);
    if (ImGui::BeginChild("##scroll", ImVec2(0, 0), ImGuiChildFlags_None,
                          ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGuiListClipper clipper;
        // Pre-filter so the clipper only iterates rows we'll actually draw.
        state.scratch_filtered.clear();
        for (size_t i = 0; i < state.scratch_entries.size(); ++i) {
            const auto& e = state.scratch_entries[i];
            if (static_cast<u32>(e.lvl) < static_cast<u32>(state.min_level)) continue;
            if (state.filter.IsActive() &&
                !state.filter.PassFilter(e.category.c_str()) &&
                !state.filter.PassFilter(e.message.c_str())) continue;
            state.scratch_filtered.push_back(i);
        }
        clipper.Begin(static_cast<int>(state.scratch_filtered.size()));
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                const auto& e  = state.scratch_entries[state.scratch_filtered[row]];
                const auto  st = level_style(e.lvl);
                ImGui::PushStyleColor(ImGuiCol_Text, st.col);
                ImGui::Text("%s", st.tag);
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, 6.0f);
                ImGui::TextDisabled("%-10s", e.category.c_str());
                ImGui::SameLine(0.0f, 6.0f);
                ImGui::TextUnformatted(e.message.c_str());
            }
        }
        if (state.auto_scroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
            ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
    ImGui::End();
}

}  // namespace cardinal::ui::panels::log_panel
