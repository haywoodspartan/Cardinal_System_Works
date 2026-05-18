// =============================================================================
// Studio — Sequencer panel implementation.
// =============================================================================
#include "sequencer.hpp"

#include <cardinal/cine/cine.hpp>

#include <imgui.h>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/cstdio.hpp>
#include <cardinal/core/traits.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::ui::panels::sequencer_panel {

namespace {

ImU32 track_color(usize idx) {
    const ImU32 palette[] = {
        IM_COL32(110, 180, 230, 220),
        IM_COL32(220, 160, 100, 220),
        IM_COL32(180, 220, 110, 220),
        IM_COL32(220, 110, 200, 220),
    };
    return palette[idx % (sizeof(palette) / sizeof(palette[0]))];
}

void draw_track_row(cardinal::cine::Sequence& s, usize ti,
                    float track_y, float row_h,
                    float t_min, float t_max,
                    const ImVec2& origin, float width)
{
    const float x_per_s = width / cardinal::max(1e-3f, t_max - t_min);
    auto x_of = [&](float t){ return origin.x + (t - t_min) * x_per_s; };

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRect({origin.x, track_y}, {origin.x + width, track_y + row_h},
                IM_COL32(60, 60, 60, 255));

    auto& track = s.tracks[ti];
    cardinal::visit([&](auto& tr) {
        using TT = cardinal::decay_t<decltype(tr)>;
        const ImU32 col = track_color(ti);
        if constexpr (cardinal::is_same_v<TT, cardinal::cine::AnimationTrack>) {
            const float x0 = x_of(tr.start_time_s);
            const float x1 = x_of(tr.start_time_s + tr.length_s);
            dl->AddRectFilled({x0, track_y + 4.0f}, {x1, track_y + row_h - 4.0f},
                              col, 4.0f);
            char lbl[64]; cardinal::snprintf(lbl, sizeof(lbl), "%s", tr.name.c_str());
            dl->AddText({x0 + 4.0f, track_y + 6.0f}, IM_COL32(20, 20, 20, 255), lbl);
        } else if constexpr (cardinal::is_same_v<TT, cardinal::cine::CameraTrack>) {
            for (const auto& sh : tr.shots) {
                const float fx = x_of(sh.time_s);
                dl->AddTriangleFilled(
                    {fx,       track_y + 4.0f},
                    {fx + 8.0f, track_y + row_h * 0.5f},
                    {fx,       track_y + row_h - 4.0f},
                    col);
            }
        } else if constexpr (cardinal::is_same_v<TT, cardinal::cine::AudioTrack>) {
            for (const auto& ev : tr.events) {
                const float fx = x_of(ev.time_s);
                dl->AddCircleFilled({fx, track_y + row_h * 0.5f}, 5.0f, col);
            }
        } else if constexpr (cardinal::is_same_v<TT, cardinal::cine::EventTrack>) {
            for (const auto& ev : tr.events) {
                const float fx = x_of(ev.time_s);
                dl->AddRectFilled({fx - 4.0f, track_y + 4.0f},
                                  {fx + 4.0f, track_y + row_h - 4.0f}, col);
            }
        }
    }, track);
}

}  // namespace

void draw(cardinal::cine::Player* player, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Sequencer", p_open,
                      ImGuiWindowFlags_NoScrollbar))
    { ImGui::End(); return; }

    if (player == nullptr || player->sequence() == nullptr) {
        ImGui::TextDisabled("(no sequence bound)");
        ImGui::End();
        return;
    }

    auto& s = *player->sequence();

    // Transport.
    if (s.playing) {
        if (ImGui::Button("Pause")) player->pause();
    } else {
        if (ImGui::Button("Play"))  player->play();
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop"))      player->stop();
    ImGui::SameLine();
    if (ImGui::Button("|<"))        player->seek(0.0f);
    ImGui::SameLine();
    ImGui::Checkbox("Loop", &s.looping);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderFloat("Speed", &s.speed, 0.0f, 4.0f, "%.2fx");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::DragFloat("Length",  &s.duration_s, 0.1f, 0.1f, 600.0f, "%.1fs");
    ImGui::SameLine();
    ImGui::Text("[%.2fs / %.2fs]", s.play_head_s, s.duration_s);

    ImGui::Separator();

    // Timeline area.
    const float t_min = 0.0f;
    const float t_max = cardinal::max(0.1f, s.duration_s);
    const float row_h = 28.0f;
    const float labels_w = 110.0f;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float full_w  = cardinal::max(200.0f, ImGui::GetContentRegionAvail().x);
    const float track_w = full_w - labels_w;
    const ImVec2 timeline_origin{ origin.x + labels_w, origin.y + 22.0f };

    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Time axis ticks.
    {
        const float x_per_s = track_w / (t_max - t_min);
        const float tick_step = (t_max < 5.0f) ? 0.5f : (t_max < 30.0f) ? 1.0f : 5.0f;
        for (float t = 0.0f; t <= t_max + 1e-3f; t += tick_step) {
            const float fx = timeline_origin.x + (t - t_min) * x_per_s;
            dl->AddLine({fx, origin.y + 4.0f}, {fx, origin.y + 18.0f},
                        IM_COL32(160, 160, 160, 255));
            char lbl[16]; cardinal::snprintf(lbl, sizeof(lbl), "%.1fs", t);
            dl->AddText({fx + 2.0f, origin.y + 4.0f}, IM_COL32(160, 160, 160, 255), lbl);
        }
    }

    // Tracks.
    for (usize i = 0; i < s.tracks.size(); ++i) {
        const float track_y = timeline_origin.y + i * (row_h + 4.0f);
        // Label column.
        cardinal::visit([&](const auto& tr) {
            char lbl[64]; cardinal::snprintf(lbl, sizeof(lbl), "%s", tr.name.c_str());
            dl->AddText({origin.x + 4.0f, track_y + 6.0f},
                        IM_COL32(220, 220, 220, 255), lbl);
        }, s.tracks[i]);
        draw_track_row(s, i, track_y, row_h, t_min, t_max,
                       timeline_origin, track_w);
    }

    // Markers.
    for (const auto& m : s.markers) {
        const float fx = timeline_origin.x + (m.time_s / t_max) * track_w;
        const float total_h = s.tracks.size() * (row_h + 4.0f);
        dl->AddLine({fx, timeline_origin.y}, {fx, timeline_origin.y + total_h},
                    IM_COL32(220, 220, 100, 255), 1.0f);
        dl->AddText({fx + 2.0f, timeline_origin.y - 14.0f},
                    IM_COL32(220, 220, 100, 255), m.label.c_str());
    }

    // Playhead.
    {
        const float fx = timeline_origin.x + (s.play_head_s / t_max) * track_w;
        const float total_h = cardinal::max(20.0f, s.tracks.size() * (row_h + 4.0f) + 6.0f);
        dl->AddLine({fx, timeline_origin.y - 6.0f},
                    {fx, timeline_origin.y + total_h},
                    IM_COL32(255, 60, 60, 255), 2.0f);
    }

    // Reserve the space we drew into so subsequent ImGui items lay out below.
    const float reserve_h = cardinal::max(60.0f,
        s.tracks.size() * (row_h + 4.0f) + 30.0f);
    ImGui::Dummy(ImVec2(full_w, reserve_h));

    // Click to scrub.
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        const float mx = ImGui::GetIO().MousePos.x;
        if (mx > timeline_origin.x) {
            const float t = (mx - timeline_origin.x) / track_w * t_max;
            player->seek(cardinal::clamp(t, 0.0f, s.duration_s));
        }
    }

    if (ImGui::CollapsingHeader("Add tracks", 0)) {
        if (ImGui::Button("+ Animation")) {
            cardinal::cine::AnimationTrack at; at.name = "Animation";
            at.length_s = s.duration_s; s.tracks.emplace_back(cardinal::move(at));
        }
        ImGui::SameLine();
        if (ImGui::Button("+ Camera")) {
            cardinal::cine::CameraTrack ct; ct.name = "Camera";
            s.tracks.emplace_back(cardinal::move(ct));
        }
        ImGui::SameLine();
        if (ImGui::Button("+ Audio")) {
            cardinal::cine::AudioTrack at; at.name = "Audio";
            s.tracks.emplace_back(cardinal::move(at));
        }
        ImGui::SameLine();
        if (ImGui::Button("+ Event")) {
            cardinal::cine::EventTrack et; et.name = "Event";
            s.tracks.emplace_back(cardinal::move(et));
        }
    }
    if (ImGui::CollapsingHeader("Markers", 0)) {
        static char mb[32] = "marker";
        ImGui::InputText("##mb", mb, sizeof(mb));
        ImGui::SameLine();
        if (ImGui::Button("Add at playhead")) {
            cardinal::cine::Marker m{}; m.label = mb;
            m.time_s = s.play_head_s; s.markers.push_back(cardinal::move(m));
        }
        for (auto it = s.markers.begin(); it != s.markers.end(); ) {
            ImGui::PushID(&*it);
            char lbl[64];
            cardinal::snprintf(lbl, sizeof(lbl), "%.2fs  %s", it->time_s, it->label.c_str());
            ImGui::TextUnformatted(lbl);
            ImGui::SameLine();
            if (ImGui::SmallButton("x")) { it = s.markers.erase(it); ImGui::PopID(); continue; }
            ImGui::PopID();
            ++it;
        }
    }
    if (ImGui::CollapsingHeader("Render config", 0)) {
        ImGui::InputText("Output", s.render.output_path.data(),
                         s.render.output_path.capacity());
        ImGui::DragFloat("Start", &s.render.start_time_s, 0.05f, 0.0f, s.duration_s, "%.2f");
        ImGui::DragFloat("End",   &s.render.end_time_s,   0.05f, 0.0f, s.duration_s, "%.2f");
        int fps = static_cast<int>(s.render.fps);
        if (ImGui::SliderInt("FPS", &fps, 1, 240)) s.render.fps = static_cast<cardinal::u32>(fps);
        int wh[2] = { static_cast<int>(s.render.width), static_cast<int>(s.render.height) };
        if (ImGui::DragInt2("Resolution", wh, 8, 64, 8192)) {
            s.render.width  = static_cast<cardinal::u32>(wh[0]);
            s.render.height = static_cast<cardinal::u32>(wh[1]);
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::sequencer_panel
