// =============================================================================
// Studio — Actor Outliner / Inspector implementation.
// =============================================================================
#include "actor_outliner.hpp"

#include <cardinal/actor/world.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/cstring.hpp>

namespace cardinal::ui::panels::actor_outliner_panel {

namespace {

// Each inspect_* returns whether it EDITED the component this frame, so the
// host can bump the World revision (driving auto-checkpoint undo).
bool inspect_transform(cardinal::actor::TransformComponent& t) {
    bool e = false;
    e |= ImGui::DragFloat3("Translation", &t.translation.x, 0.05f);
    cardinal::scene::Vec3 deg{
        t.rotation_euler.x * 57.2957795f,
        t.rotation_euler.y * 57.2957795f,
        t.rotation_euler.z * 57.2957795f
    };
    if (ImGui::DragFloat3("Rotation (deg)", &deg.x, 1.0f)) {
        t.rotation_euler.x = deg.x * 0.01745329f;
        t.rotation_euler.y = deg.y * 0.01745329f;
        t.rotation_euler.z = deg.z * 0.01745329f;
        e = true;
    }
    e |= ImGui::DragFloat3("Scale", &t.scale.x, 0.01f, 0.01f, 100.0f);
    return e;
}

bool inspect_mesh(cardinal::actor::MeshComponent& m) {
    bool e = false;
    char buf[128];
    cardinal::snprintf(buf, sizeof(buf), "%s", m.asset_id.c_str());
    if (ImGui::InputText("Asset", buf, sizeof(buf))) { m.asset_id = buf; e = true; }
    e |= ImGui::ColorEdit3("Tint", &m.tint.x);
    e |= ImGui::Checkbox("Visible", &m.visible);
    return e;
}

bool inspect_camera(cardinal::actor::CameraComponent& c) {
    bool e = false;
    float fov_deg = c.fov_y_rad * 57.2957795f;
    if (ImGui::SliderFloat("FOV (deg)", &fov_deg, 10.0f, 170.0f, "%.1f")) {
        c.fov_y_rad = fov_deg * 0.01745329f; e = true;
    }
    e |= ImGui::SliderFloat("Near", &c.z_near, 0.001f, 10.0f, "%.3f");
    e |= ImGui::SliderFloat("Far",  &c.z_far, 1.0f, 5000.0f, "%.1f");
    e |= ImGui::Checkbox("Active", &c.active);
    return e;
}

bool inspect_light(cardinal::actor::LightComponent& l) {
    bool e = false;
    const char* kinds[] = { "Directional", "Point", "Spot" };
    int k = static_cast<int>(l.kind);
    if (ImGui::Combo("Kind", &k, kinds, IM_ARRAYSIZE(kinds))) {
        l.kind = static_cast<cardinal::actor::LightKind>(k); e = true;
    }
    e |= ImGui::ColorEdit3("Color", &l.color.x);
    e |= ImGui::SliderFloat("Intensity", &l.intensity, 0.0f, 100.0f, "%.2f");
    if (l.kind != cardinal::actor::LightKind::Directional) {
        e |= ImGui::SliderFloat("Range", &l.range, 0.1f, 500.0f, "%.1f");
    }
    if (l.kind == cardinal::actor::LightKind::Spot) {
        e |= ImGui::SliderFloat("Inner cos", &l.spot_inner_cos, -1.0f, 1.0f, "%.3f");
        e |= ImGui::SliderFloat("Outer cos", &l.spot_outer_cos, -1.0f, 1.0f, "%.3f");
    }
    return e;
}

bool inspect_audio(cardinal::actor::AudioEmitterComponent& a) {
    bool e = false;
    char buf[128];
    cardinal::snprintf(buf, sizeof(buf), "%s", a.cue_id.c_str());
    if (ImGui::InputText("Cue", buf, sizeof(buf))) { a.cue_id = buf; e = true; }
    e |= ImGui::SliderFloat("Volume", &a.volume, 0.0f, 4.0f, "%.2f");
    e |= ImGui::SliderFloat("Pitch",  &a.pitch,  0.05f, 8.0f, "%.2f");
    e |= ImGui::Checkbox("Loop", &a.loop); ImGui::SameLine();
    e |= ImGui::Checkbox("3D",   &a.is_3d);
    e |= ImGui::Checkbox("Play on spawn", &a.play_on_spawn);
    int ch = static_cast<int>(a.channel);
    const char* names[] = { "Master", "Music", "SFX", "Voice", "UI" };
    if (ImGui::Combo("Channel", &ch, names, IM_ARRAYSIZE(names))) {
        a.channel = static_cast<cardinal::u32>(ch); e = true;
    }
    ImGui::TextDisabled("playing: %s, instance %llu",
        a.playing ? "yes" : "no",
        static_cast<unsigned long long>(a.instance_id));
    return e;
}

bool inspect_rb(cardinal::actor::RigidBodyComponent& r) {
    bool e = false;
    e |= ImGui::DragFloat3("Velocity",     &r.velocity.x,     0.05f);
    e |= ImGui::DragFloat3("Acceleration", &r.acceleration.x, 0.05f);
    e |= ImGui::SliderFloat("Mass",          &r.mass,            0.01f, 1000.0f, "%.2f");
    e |= ImGui::SliderFloat("Linear damping",&r.linear_damping,  0.0f,  2.0f,    "%.3f");
    e |= ImGui::Checkbox("Use gravity", &r.use_gravity); ImGui::SameLine();
    e |= ImGui::Checkbox("Kinematic",   &r.kinematic);
    return e;
}

bool inspect_tags(cardinal::actor::TagComponent& t) {
    bool e = false;
    static char nb[64] = "";
    if (ImGui::InputTextWithHint("##new_tag", "new tag", nb, sizeof(nb),
                                 ImGuiInputTextFlags_EnterReturnsTrue))
    {
        if (nb[0]) { t.add(nb); nb[0] = '\0'; e = true; }
    }
    ImGui::SameLine();
    if (ImGui::Button("Add")) { if (nb[0]) { t.add(nb); nb[0] = '\0'; e = true; } }
    for (auto it = t.tags.begin(); it != t.tags.end(); ) {
        ImGui::PushID(&*it);
        if (ImGui::SmallButton("x")) { it = t.tags.erase(it); ImGui::PopID(); e = true; continue; }
        ImGui::SameLine();
        ImGui::TextUnformatted(it->c_str());
        ImGui::PopID();
        ++it;
    }
    return e;
}

bool inspect_component(cardinal::actor::Component& c) {
    using namespace cardinal::actor;
    if (auto* p = dynamic_cast<TransformComponent*>(&c))    return inspect_transform(*p);
    if (auto* p = dynamic_cast<MeshComponent*>(&c))         return inspect_mesh(*p);
    if (auto* p = dynamic_cast<CameraComponent*>(&c))       return inspect_camera(*p);
    if (auto* p = dynamic_cast<LightComponent*>(&c))        return inspect_light(*p);
    if (auto* p = dynamic_cast<AudioEmitterComponent*>(&c)) return inspect_audio(*p);
    if (auto* p = dynamic_cast<RigidBodyComponent*>(&c))    return inspect_rb(*p);
    if (auto* p = dynamic_cast<TagComponent*>(&c))          return inspect_tags(*p);
    ImGui::TextDisabled("(no inspector)");
    return false;
}

}  // namespace

void draw(cardinal::actor::World* world,
          cardinal::u32* selected_actor_id_inout,
          const char* title, bool* p_open)
{
    if (!ImGui::Begin(title ? title : "Actors", p_open,
                      0)) { ImGui::End(); return; }
    if (world == nullptr) {
        ImGui::TextDisabled("(no actor::World bound)");
        ImGui::End();
        return;
    }

    cardinal::u32 sel = selected_actor_id_inout ? *selected_actor_id_inout : 0u;

    if (ImGui::CollapsingHeader("Spawn from blueprint", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto bps = world->blueprint_names();
        if (bps.empty()) {
            ImGui::TextDisabled("(no blueprints registered)");
        } else {
            for (const auto& bp : bps) {
                if (ImGui::SmallButton(bp.c_str())) {
                    if (auto* a = world->spawn_blueprint(bp)) {
                        sel = a->id();
                    }
                }
                ImGui::SameLine();
            }
            ImGui::NewLine();
        }
    }

    // ---- Search / filter ---------------------------------------------
    // Plain text filters by name substring (case-insensitive); a "tag:foo"
    // prefix filters by tag. Backed by the World query API so the match
    // rule is the engine's, not a UI re-implementation.
    static char s_search[128] = "";
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint("##actor_search", "filter by name, or tag:foo",
                             s_search, sizeof(s_search));
    const bool filtering = s_search[0] != '\0';

    cardinal::vector<cardinal::u32> match_ids;
    if (filtering) {
        cardinal::vector<cardinal::actor::Actor*> matches;
        if (cardinal::strncmp(s_search, "tag:", 4) == 0) {
            matches = world->find_all_by_tag(s_search + 4);
        } else {
            matches = world->find_all_by_name(s_search);
        }
        match_ids.reserve(matches.size());
        for (auto* m : matches) match_ids.push_back(m->id());
        cardinal::sort(match_ids.begin(), match_ids.end());
        ImGui::Text("Actors: %zu shown / %zu total",
                    match_ids.size(), world->actor_count());

        // ---- Bulk actions on the filtered set ------------------------
        if (!match_ids.empty()) {
            if (ImGui::SmallButton("Enable all"))  world->bulk_set_enabled(match_ids, true);
            ImGui::SameLine();
            if (ImGui::SmallButton("Disable all")) world->bulk_set_enabled(match_ids, false);
            ImGui::SameLine();
            if (ImGui::SmallButton("Delete all")) {
                world->bulk_destroy(match_ids);
                if (sel != 0 && world->find(sel) == nullptr) sel = 0;   // cleared if gone
            }
            // Bulk tag add/remove on the shown set.
            static char s_bulk_tag[64] = "";
            ImGui::SetNextItemWidth(120.0f);
            ImGui::InputTextWithHint("##bulk_tag", "tag", s_bulk_tag, sizeof(s_bulk_tag));
            ImGui::SameLine();
            const bool has_tag = s_bulk_tag[0] != '\0';
            if (!has_tag) ImGui::BeginDisabled();
            if (ImGui::SmallButton("+Tag")) world->bulk_add_tag(match_ids, s_bulk_tag);
            ImGui::SameLine();
            if (ImGui::SmallButton("-Tag")) world->bulk_remove_tag(match_ids, s_bulk_tag);
            if (!has_tag) ImGui::EndDisabled();

            // Align (to set center) + distribute (evenly) along an axis.
            using Ax = cardinal::actor::World::Axis;
            using Mode = cardinal::actor::World::AlignMode;
            ImGui::TextUnformatted("Align:");
            ImGui::SameLine();
            if (ImGui::SmallButton("X")) world->align_actors(match_ids, Ax::X, Mode::Center);
            ImGui::SameLine();
            if (ImGui::SmallButton("Y")) world->align_actors(match_ids, Ax::Y, Mode::Center);
            ImGui::SameLine();
            if (ImGui::SmallButton("Z")) world->align_actors(match_ids, Ax::Z, Mode::Center);
            ImGui::SameLine();
            ImGui::TextUnformatted(" Distribute:");
            ImGui::SameLine();
            if (ImGui::SmallButton("X##d")) world->distribute_actors(match_ids, Ax::X);
            ImGui::SameLine();
            if (ImGui::SmallButton("Y##d")) world->distribute_actors(match_ids, Ax::Y);
            ImGui::SameLine();
            if (ImGui::SmallButton("Z##d")) world->distribute_actors(match_ids, Ax::Z);
        }
    } else {
        ImGui::Text("Actors: %zu", world->actor_count());
    }
    ImGui::Separator();

    const float h = cardinal::max(120.0f, ImGui::GetContentRegionAvail().y * 0.45f);
    if (ImGui::BeginChild("##actor_list", ImVec2(0, h), ImGuiChildFlags_FrameStyle))
    {
        for (const auto& a : world->actors()) {
            if (!a->alive()) continue;
            if (filtering &&
                !cardinal::binary_search(match_ids.begin(), match_ids.end(), a->id()))
                continue;
            ImGui::PushID(static_cast<int>(a->id()));
            // Per-row active checkbox — toggles the actor on/off in the sim
            // without deleting it.
            bool en = a->enabled();
            if (ImGui::Checkbox("##en", &en)) { a->set_enabled(en); world->bump_revision(); }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(en ? "Active — click to disable"
                                     : "Disabled — click to enable");
            ImGui::SameLine();
            char label[128];
            cardinal::snprintf(label, sizeof(label), "[%u] %s",
                a->id(), a->name().c_str());
            // Dim disabled actors so the scene state reads at a glance.
            if (!en) ImGui::PushStyleColor(ImGuiCol_Text,
                                           ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
            if (ImGui::Selectable(label, sel == a->id())) sel = a->id();
            if (!en) ImGui::PopStyleColor();
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem(a->enabled() ? "Disable" : "Enable")) {
                    a->set_enabled(!a->enabled());
                    world->bump_revision();
                }
                if (ImGui::MenuItem("Duplicate")) {
                    if (auto* dup = world->duplicate(a->id())) sel = dup->id();
                }
                if (ImGui::MenuItem("Destroy")) world->destroy(a->id());
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
    }
    ImGui::EndChild();

    if (selected_actor_id_inout) *selected_actor_id_inout = sel;

    ImGui::Separator();
    ImGui::Text("Inspector");

    cardinal::actor::Actor* a = world->find(sel);
    if (a == nullptr) {
        ImGui::TextDisabled("(no actor selected)");
    } else {
        char nm[128];
        cardinal::snprintf(nm, sizeof(nm), "%s", a->name().c_str());
        if (ImGui::InputText("Name", nm, sizeof(nm))) { a->set_name(nm); world->bump_revision(); }
        ImGui::Text("ID:     %u", a->id());
        ImGui::Text("Parent: %u", a->parent());
        ImGui::Separator();
        // Inter-frame component clipboard (one Inspector per Studio, so a
        // function-static is fine — matches save_load's static path fields).
        static cardinal::string s_clipboard;
        static cardinal::string s_clip_type;

        // Per-component headers with Copy + remove (X) affordances. Transform
        // is not removable — every actor needs one. Defer removal until after
        // the loop so we don't mutate the vector mid-iteration.
        const char* to_remove = nullptr;
        for (const auto& c : a->components()) {
            ImGui::PushID(c.get());
            const char* ctype = c->type_name();
            // Copy this component's values to the clipboard.
            if (ImGui::SmallButton("Copy")) {
                s_clipboard = cardinal::actor::copy_component(*c);
                s_clip_type = ctype;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Copy this %s component's values", ctype);
            ImGui::SameLine();
            const bool removable = cardinal::strcmp(ctype, "Transform") != 0;
            if (removable) {
                if (ImGui::SmallButton("X")) to_remove = ctype;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Remove this %s component", ctype);
                ImGui::SameLine();
            }
            if (ImGui::CollapsingHeader(ctype, ImGuiTreeNodeFlags_DefaultOpen)) {
                // Field edits bump the World revision -> Studio auto-captures
                // an undo checkpoint once the edit settles.
                if (inspect_component(*c)) world->bump_revision();
            }
            ImGui::PopID();
        }
        if (to_remove != nullptr) { a->remove_component(to_remove); world->bump_revision(); }

        // Paste the clipboard onto this actor (overwrites the matching
        // component's values, or adds one if absent).
        if (!s_clipboard.empty()) {
            char plabel[96];
            cardinal::snprintf(plabel, sizeof(plabel), "Paste %s", s_clip_type.c_str());
            if (ImGui::Button(plabel)) {
                cardinal::actor::paste_component(*a, s_clipboard);
                world->bump_revision();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Apply the copied %s values to this actor "
                                  "(adds the component if missing)",
                                  s_clip_type.c_str());
        }

        ImGui::Separator();
        // Add Component — data-driven over the built-in factory. Each type
        // greys out once present (add_component does NOT dedup, and the
        // get_component query only ever returns the first of a type, so a
        // second copy would be dead weight). Uses make_component_by_name +
        // adopt_component so the set stays in sync with the persistence
        // factory automatically.
        ImGui::TextUnformatted("Add Component:");
        static const char* kAddable[] = {
            "Mesh", "Camera", "Light", "AudioEmitter",
            "RigidBody", "Tag", "Script", "PlayerController",
        };
        int col = 0;
        for (const char* type : kAddable) {
            const bool present = a->has_component(type);
            if (present) ImGui::BeginDisabled();
            char label[64];
            cardinal::snprintf(label, sizeof(label), "+ %s", type);
            if (ImGui::Button(label)) {
                if (auto comp = cardinal::actor::make_component_by_name(type)) {
                    a->adopt_component(cardinal::move(comp));
                    world->bump_revision();
                }
            }
            if (present) ImGui::EndDisabled();
            // 3 buttons per row.
            if (++col % 3 != 0) ImGui::SameLine();
        }
        if (col % 3 != 0) ImGui::NewLine();
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::actor_outliner_panel
