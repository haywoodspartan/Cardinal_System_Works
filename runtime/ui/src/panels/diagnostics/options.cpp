// =============================================================================
// Studio Options / Settings panel implementation. A GUI over the console CVar
// registry — the same knobs the `set` command + config files drive — grouped
// by dotted-namespace category, with a Save/Load bar over the settings file.
// =============================================================================
#include <cardinal/ui/options_panel.hpp>

#include <cardinal/ui/imgui.hpp>
#include <cardinal/console/console.hpp>

namespace cardinal::ui::panels::options_panel {

namespace cv = cardinal::console;

namespace {

void draw_cvar(const cv::CVar* c) {
    if (c == nullptr) return;
    const char* label = c->name.c_str();
    switch (c->type) {
        case cv::CVarType::Bool: {
            bool b = c->get_bool ? c->get_bool() : false;
            if (ImGui::Checkbox(label, &b) && c->set_bool) c->set_bool(b);
            break;
        }
        case cv::CVarType::Int: {
            int v = c->get_int ? static_cast<int>(c->get_int()) : 0;
            const int lo = static_cast<int>(c->min);
            const int hi = static_cast<int>(c->max);
            if (hi > lo) {
                if (ImGui::SliderInt(label, &v, lo, hi) && c->set_int)
                    c->set_int(static_cast<cardinal::i64>(v));
            } else if (ImGui::InputInt(label, &v) && c->set_int) {
                c->set_int(static_cast<cardinal::i64>(v));
            }
            break;
        }
        case cv::CVarType::Float: {
            float v = c->get_float ? static_cast<float>(c->get_float()) : 0.0f;
            const float lo = static_cast<float>(c->min);
            const float hi = static_cast<float>(c->max);
            if (hi > lo) {
                if (ImGui::SliderFloat(label, &v, lo, hi) && c->set_float)
                    c->set_float(static_cast<double>(v));
            } else if (ImGui::InputFloat(label, &v) && c->set_float) {
                c->set_float(static_cast<double>(v));
            }
            break;
        }
        case cv::CVarType::String: {
            const cardinal::string s = c->get_string ? c->get_string()
                                                     : cardinal::string{};
            ImGui::LabelText(label, "%s", s.c_str());   // read-only; edit via console
            break;
        }
    }
    if (ImGui::IsItemHovered() && !c->help.empty())
        ImGui::SetTooltip("%s", c->help.c_str());
}

}  // namespace

void draw(const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Options", p_open, 0)) { ImGui::End(); return; }

    cv::Registry& reg = cv::Registry::instance();

    // ---- Save / Load bar over the settings file --------------------------
    static char path[256] = "studio_settings.cfg";
    ImGui::SetNextItemWidth(220.0f);
    ImGui::InputText("##cfgpath", path, sizeof(path));
    ImGui::SameLine();
    if (ImGui::Button("Save")) {
        const auto n = reg.save_cvars(path);
        (void)n;
    }
    ImGui::SameLine();
    if (ImGui::Button("Load")) {
        cv::Output sink{[](const cardinal::string&) {}};
        reg.exec_file(path, sink);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("settings file");

    static char filter[64] = "";
    ImGui::SetNextItemWidth(220.0f);
    ImGui::InputText("Filter", filter, sizeof(filter));
    ImGui::Separator();

    // all_cvars() is sorted by name, so same-prefix cvars are contiguous —
    // group them under a collapsing header per dotted-namespace category.
    const auto cvars = reg.all_cvars();
    cardinal::string cur_cat;
    bool cat_open = false;
    int  shown    = 0;
    for (const cv::CVar* c : cvars) {
        if (c == nullptr) continue;
        if (filter[0] != '\0' &&
            c->name.find(filter) == cardinal::string::npos) continue;   // name filter
        const auto dot = c->name.find('.');
        const cardinal::string cat = (dot == cardinal::string::npos)
            ? cardinal::string("general") : c->name.substr(0, dot);
        if (cat != cur_cat) {
            cur_cat  = cat;
            cat_open = ImGui::CollapsingHeader(cur_cat.c_str(),
                                               ImGuiTreeNodeFlags_DefaultOpen);
        }
        if (cat_open) { draw_cvar(c); ++shown; }
    }
    if (shown == 0) ImGui::TextDisabled("No settings match the filter.");

    ImGui::End();
}

}  // namespace cardinal::ui::panels::options_panel
