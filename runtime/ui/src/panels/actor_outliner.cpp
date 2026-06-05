// =============================================================================
// Studio — Actor Outliner / Inspector implementation.
// =============================================================================
#include "actor_outliner.hpp"

#include <cardinal/actor/world.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cstdio.hpp>

namespace cardinal::ui::panels::actor_outliner_panel {

namespace {

void inspect_transform(cardinal::actor::TransformComponent& t) {
    ImGui::DragFloat3("Translation", &t.translation.x, 0.05f);
    cardinal::scene::Vec3 deg{
        t.rotation_euler.x * 57.2957795f,
        t.rotation_euler.y * 57.2957795f,
        t.rotation_euler.z * 57.2957795f
    };
    if (ImGui::DragFloat3("Rotation (deg)", &deg.x, 1.0f)) {
        t.rotation_euler.x = deg.x * 0.01745329f;
        t.rotation_euler.y = deg.y * 0.01745329f;
        t.rotation_euler.z = deg.z * 0.01745329f;
    }
    ImGui::DragFloat3("Scale", &t.scale.x, 0.01f, 0.01f, 100.0f);
}

void inspect_mesh(cardinal::actor::MeshComponent& m) {
    char buf[128];
    cardinal::snprintf(buf, sizeof(buf), "%s", m.asset_id.c_str());
    if (ImGui::InputText("Asset", buf, sizeof(buf))) m.asset_id = buf;
    ImGui::ColorEdit3("Tint", &m.tint.x);
    ImGui::Checkbox("Visible", &m.visible);
}

void inspect_camera(cardinal::actor::CameraComponent& c) {
    float fov_deg = c.fov_y_rad * 57.2957795f;
    if (ImGui::SliderFloat("FOV (deg)", &fov_deg, 10.0f, 170.0f, "%.1f")) {
        c.fov_y_rad = fov_deg * 0.01745329f;
    }
    ImGui::SliderFloat("Near", &c.z_near, 0.001f, 10.0f, "%.3f");
    ImGui::SliderFloat("Far",  &c.z_far, 1.0f, 5000.0f, "%.1f");
    ImGui::Checkbox("Active", &c.active);
}

void inspect_light(cardinal::actor::LightComponent& l) {
    const char* kinds[] = { "Directional", "Point", "Spot" };
    int k = static_cast<int>(l.kind);
    if (ImGui::Combo("Kind", &k, kinds, IM_ARRAYSIZE(kinds))) {
        l.kind = static_cast<cardinal::actor::LightKind>(k);
    }
    ImGui::ColorEdit3("Color", &l.color.x);
    ImGui::SliderFloat("Intensity", &l.intensity, 0.0f, 100.0f, "%.2f");
    if (l.kind != cardinal::actor::LightKind::Directional) {
        ImGui::SliderFloat("Range", &l.range, 0.1f, 500.0f, "%.1f");
    }
    if (l.kind == cardinal::actor::LightKind::Spot) {
        ImGui::SliderFloat("Inner cos", &l.spot_inner_cos, -1.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("Outer cos", &l.spot_outer_cos, -1.0f, 1.0f, "%.3f");
    }
}

void inspect_audio(cardinal::actor::AudioEmitterComponent& a) {
    char buf[128];
    cardinal::snprintf(buf, sizeof(buf), "%s", a.cue_id.c_str());
    if (ImGui::InputText("Cue", buf, sizeof(buf))) a.cue_id = buf;
    ImGui::SliderFloat("Volume", &a.volume, 0.0f, 4.0f, "%.2f");
    ImGui::SliderFloat("Pitch",  &a.pitch,  0.05f, 8.0f, "%.2f");
    ImGui::Checkbox("Loop", &a.loop); ImGui::SameLine();
    ImGui::Checkbox("3D",   &a.is_3d);
    ImGui::Checkbox("Play on spawn", &a.play_on_spawn);
    int ch = static_cast<int>(a.channel);
    const char* names[] = { "Master", "Music", "SFX", "Voice", "UI" };
    if (ImGui::Combo("Channel", &ch, names, IM_ARRAYSIZE(names))) {
        a.channel = static_cast<cardinal::u32>(ch);
    }
    ImGui::TextDisabled("playing: %s, instance %llu",
        a.playing ? "yes" : "no",
        static_cast<unsigned long long>(a.instance_id));
}

void inspect_rb(cardinal::actor::RigidBodyComponent& r) {
    ImGui::DragFloat3("Velocity",     &r.velocity.x,     0.05f);
    ImGui::DragFloat3("Acceleration", &r.acceleration.x, 0.05f);
    ImGui::SliderFloat("Mass",          &r.mass,            0.01f, 1000.0f, "%.2f");
    ImGui::SliderFloat("Linear damping",&r.linear_damping,  0.0f,  2.0f,    "%.3f");
    ImGui::Checkbox("Use gravity", &r.use_gravity); ImGui::SameLine();
    ImGui::Checkbox("Kinematic",   &r.kinematic);
}

void inspect_tags(cardinal::actor::TagComponent& t) {
    static char nb[64] = "";
    if (ImGui::InputTextWithHint("##new_tag", "new tag", nb, sizeof(nb),
                                 ImGuiInputTextFlags_EnterReturnsTrue))
    {
        if (nb[0]) { t.add(nb); nb[0] = '\0'; }
    }
    ImGui::SameLine();
    if (ImGui::Button("Add")) { if (nb[0]) { t.add(nb); nb[0] = '\0'; } }
    for (auto it = t.tags.begin(); it != t.tags.end(); ) {
        ImGui::PushID(&*it);
        if (ImGui::SmallButton("x")) { it = t.tags.erase(it); ImGui::PopID(); continue; }
        ImGui::SameLine();
        ImGui::TextUnformatted(it->c_str());
        ImGui::PopID();
        ++it;
    }
}

void inspect_component(cardinal::actor::Component& c) {
    using namespace cardinal::actor;
    if (auto* p = dynamic_cast<TransformComponent*>(&c))    { inspect_transform(*p); return; }
    if (auto* p = dynamic_cast<MeshComponent*>(&c))         { inspect_mesh(*p);      return; }
    if (auto* p = dynamic_cast<CameraComponent*>(&c))       { inspect_camera(*p);    return; }
    if (auto* p = dynamic_cast<LightComponent*>(&c))        { inspect_light(*p);     return; }
    if (auto* p = dynamic_cast<AudioEmitterComponent*>(&c)) { inspect_audio(*p);     return; }
    if (auto* p = dynamic_cast<RigidBodyComponent*>(&c))    { inspect_rb(*p);        return; }
    if (auto* p = dynamic_cast<TagComponent*>(&c))          { inspect_tags(*p);      return; }
    ImGui::TextDisabled("(no inspector)");
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

    ImGui::Text("Actors: %zu", world->actor_count());
    ImGui::Separator();

    const float h = cardinal::max(120.0f, ImGui::GetContentRegionAvail().y * 0.45f);
    if (ImGui::BeginChild("##actor_list", ImVec2(0, h), ImGuiChildFlags_FrameStyle))
    {
        for (const auto& a : world->actors()) {
            if (!a->alive()) continue;
            ImGui::PushID(static_cast<int>(a->id()));
            char label[128];
            cardinal::snprintf(label, sizeof(label), "[%u] %s",
                a->id(), a->name().c_str());
            if (ImGui::Selectable(label, sel == a->id())) sel = a->id();
            if (ImGui::BeginPopupContextItem()) {
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
        if (ImGui::InputText("Name", nm, sizeof(nm))) a->set_name(nm);
        ImGui::Text("ID:     %u", a->id());
        ImGui::Text("Parent: %u", a->parent());
        ImGui::Separator();
        for (const auto& c : a->components()) {
            ImGui::PushID(c.get());
            if (ImGui::CollapsingHeader(c->type_name(),
                                        ImGuiTreeNodeFlags_DefaultOpen)) {
                inspect_component(*c);
            }
            ImGui::PopID();
        }
        ImGui::Separator();
        if (ImGui::Button("+ Mesh"))         a->add_component<cardinal::actor::MeshComponent>();
        ImGui::SameLine();
        if (ImGui::Button("+ Camera"))       a->add_component<cardinal::actor::CameraComponent>();
        ImGui::SameLine();
        if (ImGui::Button("+ Light"))        a->add_component<cardinal::actor::LightComponent>();
        if (ImGui::Button("+ AudioEmitter")) a->add_component<cardinal::actor::AudioEmitterComponent>();
        ImGui::SameLine();
        if (ImGui::Button("+ RigidBody"))    a->add_component<cardinal::actor::RigidBodyComponent>();
        ImGui::SameLine();
        if (ImGui::Button("+ Tag"))          a->add_component<cardinal::actor::TagComponent>();
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::actor_outliner_panel
