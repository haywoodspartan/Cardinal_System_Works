// =============================================================================
// Studio — Game class picker / property editor.
// =============================================================================
#include "class_picker.hpp"

#include <cardinal/game/game.hpp>
#include <cardinal/game/reflection.hpp>

#include <imgui.h>

#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/containers.hpp>

namespace cardinal::ui::panels::class_picker_panel {

namespace {

void edit_property(cardinal::game::PropertyDef& p) {
    using K = cardinal::game::PropertyKind;
    ImGui::PushID(p.name.c_str());
    switch (p.kind) {
        case K::Float: {
            float* f = static_cast<float*>(p.ptr);
            ImGui::SliderFloat(p.name.c_str(), f, p.fmin, p.fmax, "%.3f");
            break;
        }
        case K::Int: {
            int* i = static_cast<int*>(p.ptr);
            ImGui::SliderInt(p.name.c_str(), i, p.imin, p.imax);
            break;
        }
        case K::Bool: {
            bool* b = static_cast<bool*>(p.ptr);
            ImGui::Checkbox(p.name.c_str(), b);
            break;
        }
        case K::Vec3: {
            float* v = static_cast<float*>(p.ptr);
            ImGui::DragFloat3(p.name.c_str(), v, 0.05f);
            break;
        }
        case K::Color: {
            float* v = static_cast<float*>(p.ptr);
            ImGui::ColorEdit3(p.name.c_str(), v);
            break;
        }
        case K::String: {
            cardinal::string* s = static_cast<cardinal::string*>(p.ptr);
            char buf[256];
            cardinal::snprintf(buf, sizeof(buf), "%s", s->c_str());
            if (ImGui::InputText(p.name.c_str(), buf, sizeof(buf))) {
                *s = buf;
            }
            break;
        }
    }
    if (ImGui::IsItemHovered() && !p.tooltip.empty()) {
        ImGui::SetTooltip("%s", p.tooltip.c_str());
    }
    ImGui::PopID();
}

}  // namespace

void draw(cardinal::game::Game* game, cardinal::u32* selected_actor_id_inout,
          const char* title, bool* p_open)
{
    if (!ImGui::Begin(title ? title : "Game Classes", p_open)) {
        ImGui::End();
        return;
    }
    if (game == nullptr) {
        ImGui::TextDisabled("(no game::Game bound)");
        ImGui::End();
        return;
    }

    const auto& reg = cardinal::game::ClassRegistry::instance();
    const auto names = reg.all_names();

    if (ImGui::CollapsingHeader("Registered classes", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("%zu class(es) registered", names.size());
        // Group by leading category prefix (split on first "/").
        cardinal::map<cardinal::string, cardinal::vector<const cardinal::game::ClassDef*>> by_cat;
        for (const auto& n : names) {
            const auto* d = reg.find(n);
            if (d == nullptr) continue;
            cardinal::string cat = d->category;
            if (cat.empty()) cat = "(uncategorised)";
            const auto slash = cat.find('/');
            if (slash != cardinal::string::npos) cat = cat.substr(0, slash);
            by_cat[cat].push_back(d);
        }
        for (const auto& [cat, defs] : by_cat) {
            if (ImGui::TreeNodeEx(cat.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                for (const auto* d : defs) {
                    ImGui::PushID(d);
                    ImGui::TextUnformatted(d->name.c_str());
                    if (!d->category.empty() && d->category != cat) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("(%s)", d->category.c_str());
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Spawn")) {
                        if (auto* a = game->spawn_class(d->name)) {
                            if (selected_actor_id_inout) *selected_actor_id_inout = a->id();
                        }
                    }
                    ImGui::PopID();
                }
                ImGui::TreePop();
            }
        }
    }

    ImGui::Separator();
    ImGui::Text("Selected actor properties");

    const cardinal::u32 sel = selected_actor_id_inout ? *selected_actor_id_inout : 0u;
    cardinal::actor::Actor* a = game->world().find(sel);
    if (a == nullptr) {
        ImGui::TextDisabled("(no actor selected)");
        ImGui::End();
        return;
    }

    cardinal::game::GameActor* ga = nullptr;
    for (const auto& c : a->components()) {
        if (auto* g = dynamic_cast<cardinal::game::GameActor*>(c.get())) {
            ga = g; break;
        }
    }
    if (ga == nullptr) {
        ImGui::TextDisabled("(this actor has no GameActor component — spawn via the picker above)");
        ImGui::End();
        return;
    }

    ImGui::Text("Class:    %s", ga->class_name().c_str());
    ImGui::Text("Playing:  %s", ga->playing() ? "yes" : "no");

    const auto* def = reg.find(ga->class_name());
    if (def == nullptr || !def->describe_properties) {
        ImGui::TextDisabled("(no reflected properties)");
        ImGui::End();
        return;
    }

    auto props = def->describe_properties(ga);
    if (props.empty()) {
        ImGui::TextDisabled("(no exposed properties)");
    } else {
        for (auto& p : props) edit_property(p);
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::class_picker_panel
