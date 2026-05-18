// =============================================================================
// Studio — Audio Mixer panel implementation.
// =============================================================================
#include "mixer.hpp"

#include <cardinal/audio/audio.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/containers.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::ui::panels::mixer_panel {

void draw(cardinal::audio::Engine* engine, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Mixer", p_open)) { ImGui::End(); return; }
    if (engine == nullptr) {
        ImGui::TextDisabled("(no audio::Engine bound)");
        ImGui::End();
        return;
    }

    // Sum per-channel attenuated volume across active instances for a
    // VU-style meter. Cheap (we already hold the snapshot).
    cardinal::unordered_map<cardinal::audio::ChannelId, float> meter;
    {
        for (const auto& s : engine->active_instances()) {
            meter[s.channel] += s.final_attenuated_volume;
        }
    }

    const auto chans = engine->channels();
    const float strip_w = 80.0f;
    const float fader_h = 220.0f;

    if (ImGui::BeginChild("##mixer_strips",
                          ImVec2(0, fader_h + 80.0f),
                          ImGuiChildFlags_FrameStyle))
    {
        for (const auto& c : chans) {
            ImGui::PushID(static_cast<int>(c.id));
            ImGui::BeginGroup();

            ImGui::TextUnformatted(c.name.c_str());

            // Vertical fader.
            float v = c.volume;
            if (ImGui::VSliderFloat("##fader", ImVec2(strip_w * 0.6f, fader_h),
                                    &v, 0.0f, 1.0f, "%.2f")) {
                engine->set_channel_volume(c.id, v);
            }

            // Meter bar to the right of the fader.
            ImGui::SameLine();
            const float lvl = cardinal::min(1.0f, meter[c.id]);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                lvl > 0.85f ? ImVec4(0.95f, 0.30f, 0.30f, 1.0f)
              : lvl > 0.55f ? ImVec4(0.95f, 0.85f, 0.30f, 1.0f)
                            : ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
            ImGui::ProgressBar(lvl, ImVec2(strip_w * 0.25f, fader_h), "");
            ImGui::PopStyleColor();

            // Mute / solo toggles.
            bool mute = c.muted;
            if (ImGui::Checkbox("M", &mute)) engine->set_channel_muted(c.id, mute);
            ImGui::SameLine();
            bool solo = c.solo;
            if (ImGui::Checkbox("S", &solo)) {
                if (auto* ch = engine->channel(c.id)) ch->solo = solo;
            }

            ImGui::EndGroup();
            ImGui::PopID();
            ImGui::SameLine();
        }
    }
    ImGui::EndChild();

    ImGui::Separator();
    const auto stats = engine->stats();
    ImGui::Text("Cues:      %u", stats.cues_registered);
    ImGui::Text("Active:    %u (total played %llu, culled %llu)",
        stats.active_instances,
        static_cast<unsigned long long>(stats.total_played),
        static_cast<unsigned long long>(stats.total_culled));

    if (ImGui::Button("Play preview tone (440Hz)")) {
        // Register a built-in tone the first time we need it.
        if (engine->find_cue("__studio_preview_tone__") == nullptr) {
            cardinal::audio::Cue c{};
            c.id = "__studio_preview_tone__";
            c.kind = cardinal::audio::CueKind::SineWave;
            c.duration_s = 0.4f;
            c.sine_frequency_hz = 440.0f;
            c.gain = 0.6f;
            engine->register_cue(cardinal::move(c));
        }
        engine->play_2d("__studio_preview_tone__",
                        cardinal::audio::kChannelUi, 1.0f, 1.0f, false);
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::mixer_panel
