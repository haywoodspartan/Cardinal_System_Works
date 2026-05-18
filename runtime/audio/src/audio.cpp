#include <cardinal/audio/audio.hpp>

#include <cardinal/core/log.hpp>

#include <algorithm>
#include <cmath>

namespace cardinal::audio {

const char* channel_name(ChannelId id) noexcept {
    switch (id) {
        case kChannelMaster: return "Master";
        case kChannelMusic:  return "Music";
        case kChannelSfx:    return "SFX";
        case kChannelVoice:  return "Voice";
        case kChannelUi:     return "UI";
    }
    return "Channel";
}

std::shared_ptr<Engine> Engine::create(const EngineDesc& desc) {
    auto e = std::shared_ptr<Engine>(new Engine());
    if (!e->initialize_(desc)) return nullptr;
    return e;
}

Engine::~Engine() = default;

bool Engine::initialize_(const EngineDesc& desc) {
    desc_ = desc;
    // Default channel hierarchy.
    channels_[kChannelMaster] = { kChannelMaster, "Master", 1.0f, false, false, kChannelMaster };
    channels_[kChannelMusic ] = { kChannelMusic,  "Music",  0.8f, false, false, kChannelMaster };
    channels_[kChannelSfx   ] = { kChannelSfx,    "SFX",    1.0f, false, false, kChannelMaster };
    channels_[kChannelVoice ] = { kChannelVoice,  "Voice",  1.0f, false, false, kChannelMaster };
    channels_[kChannelUi    ] = { kChannelUi,     "UI",     1.0f, false, false, kChannelMaster };
    cardinal::log::infof("audio/engine",
        "Audio engine online — %u Hz, max %u instances, attenuation %.1fm..%.1fm",
        desc_.sample_rate, desc_.max_instances, desc_.distance_min, desc_.distance_max);
    return true;
}

void Engine::register_cue(Cue cue) {
    std::lock_guard<std::mutex> lg(mtx_);
    cues_[cue.id] = std::move(cue);
}
void Engine::unregister_cue(const std::string& id) {
    std::lock_guard<std::mutex> lg(mtx_);
    cues_.erase(id);
}
const Cue* Engine::find_cue(const std::string& id) const {
    std::lock_guard<std::mutex> lg(mtx_);
    auto it = cues_.find(id);
    return it == cues_.end() ? nullptr : &it->second;
}

Channel* Engine::channel(ChannelId id) {
    std::lock_guard<std::mutex> lg(mtx_);
    auto it = channels_.find(id);
    return it == channels_.end() ? nullptr : &it->second;
}
const Channel* Engine::channel(ChannelId id) const {
    return const_cast<Engine*>(this)->channel(id);
}
void Engine::set_channel_volume(ChannelId id, float v) {
    if (auto* c = channel(id)) {
        std::lock_guard<std::mutex> lg(mtx_);
        c->volume = std::clamp(v, 0.0f, 1.0f);
    }
}
void Engine::set_channel_muted(ChannelId id, bool m) {
    if (auto* c = channel(id)) {
        std::lock_guard<std::mutex> lg(mtx_);
        c->muted = m;
    }
}

bool Engine::any_solo_active_() const noexcept {
    for (const auto& [_, ch] : channels_) if (ch.solo) return true;
    return false;
}

float Engine::effective_volume(ChannelId id) const {
    std::lock_guard<std::mutex> lg(mtx_);
    const bool soloing = any_solo_active_();
    float v = 1.0f;
    ChannelId cur = id;
    while (true) {
        auto it = channels_.find(cur);
        if (it == channels_.end()) break;
        const Channel& ch = it->second;
        if (ch.muted) return 0.0f;
        if (soloing && !ch.solo && cur != kChannelMaster) return 0.0f;
        v *= ch.volume;
        if (cur == ch.parent) break;
        cur = ch.parent;
    }
    return v;
}

std::vector<Channel> Engine::channels() const {
    std::vector<Channel> r;
    std::lock_guard<std::mutex> lg(mtx_);
    r.reserve(channels_.size());
    for (const auto& [_, c] : channels_) r.push_back(c);
    std::sort(r.begin(), r.end(),
              [](const Channel& a, const Channel& b){ return a.id < b.id; });
    return r;
}

void Engine::set_listener(const Listener& l) { listener_ = l; }

float Engine::compute_3d_attenuation_(const cardinal::scene::Vec3& pos) const noexcept {
    const float dx = pos.x - listener_.position.x;
    const float dy = pos.y - listener_.position.y;
    const float dz = pos.z - listener_.position.z;
    const float d  = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (d <= desc_.distance_min) return 1.0f;
    if (d >= desc_.distance_max) return 0.0f;
    const float t = (d - desc_.distance_min) / (desc_.distance_max - desc_.distance_min);
    return std::pow(1.0f - t, std::max(0.1f, desc_.rolloff));
}

InstanceId Engine::play_2d(const std::string& cue_id, ChannelId ch,
                           float volume, float pitch, bool loop)
{
    return play_3d(cue_id, {0,0,0}, ch, volume, pitch, loop) | (1ull << 63);   // mark 2D
}

InstanceId Engine::play_3d(const std::string& cue_id, const cardinal::scene::Vec3& pos,
                           ChannelId ch, float volume, float pitch, bool loop)
{
    const Cue* c = find_cue(cue_id);
    if (c == nullptr) {
        cardinal::log::warnf("audio/engine", "play: unknown cue '%s'", cue_id.c_str());
        return 0;
    }
    std::lock_guard<std::mutex> lg(mtx_);
    if (instances_.size() >= desc_.max_instances) {
        // Cull the oldest non-looping instance.
        auto it = std::find_if(instances_.begin(), instances_.end(),
                               [](const InstanceState& s){ return !s.loop; });
        if (it != instances_.end()) instances_.erase(it);
        else if (!instances_.empty()) instances_.erase(instances_.begin());
        ++total_culled_;
    }
    InstanceState s{};
    s.id          = next_instance_id_++;
    s.cue_id      = cue_id;
    s.channel     = ch;
    s.is_3d       = true;
    s.position    = pos;
    s.volume      = std::clamp(volume, 0.0f, 4.0f);
    s.pitch       = std::clamp(pitch, 0.05f, 8.0f);
    s.loop        = loop;
    s.duration_s  = c->duration_s;
    instances_.push_back(s);
    ++total_played_;
    return s.id;
}

void Engine::fade_in(InstanceId id, float seconds) {
    std::lock_guard<std::mutex> lg(mtx_);
    for (auto& s : instances_) if (s.id == id) {
        s.fade_in_s = std::max(0.0f, seconds);
        s.play_head_s = 0.0f;
    }
}
void Engine::fade_out(InstanceId id, float seconds) {
    std::lock_guard<std::mutex> lg(mtx_);
    for (auto& s : instances_) if (s.id == id) {
        s.fade_out_s = std::max(0.0f, seconds);
        s.stopping = true;
    }
}
void Engine::stop(InstanceId id) {
    std::lock_guard<std::mutex> lg(mtx_);
    instances_.erase(std::remove_if(instances_.begin(), instances_.end(),
        [id](const InstanceState& s){ return s.id == id; }), instances_.end());
}
void Engine::set_emitter_position(InstanceId id, const cardinal::scene::Vec3& pos) {
    std::lock_guard<std::mutex> lg(mtx_);
    for (auto& s : instances_) if (s.id == id) { s.position = pos; s.is_3d = true; }
}

void Engine::tick(float dt) {
    std::lock_guard<std::mutex> lg(mtx_);
    for (auto& s : instances_) {
        s.play_head_s += dt * s.pitch;
        // Channel volume.
        float v = s.volume;
        // effective_volume already takes the lock — call the unlocked walk inline.
        const bool soloing = any_solo_active_();
        ChannelId cur = s.channel;
        while (true) {
            auto it = channels_.find(cur);
            if (it == channels_.end()) break;
            const Channel& ch = it->second;
            if (ch.muted) { v = 0.0f; break; }
            if (soloing && !ch.solo && cur != kChannelMaster) { v = 0.0f; break; }
            v *= ch.volume;
            if (cur == ch.parent) break;
            cur = ch.parent;
        }
        // 3D attenuation.
        if (s.is_3d) v *= compute_3d_attenuation_(s.position);
        // Fade in / out scaling.
        if (s.fade_in_s > 0.0f && s.play_head_s < s.fade_in_s) {
            v *= s.play_head_s / s.fade_in_s;
        }
        if (s.stopping && s.fade_out_s > 0.0f) {
            const float t = std::max(0.0f, s.fade_out_s - dt);
            s.fade_out_s = t;
            v *= t / std::max(1e-3f, s.fade_out_s + dt);
        }
        s.final_attenuated_volume = std::clamp(v, 0.0f, 1.0f);
    }
    // Remove completed / stopped.
    instances_.erase(std::remove_if(instances_.begin(), instances_.end(),
        [](const InstanceState& s){
            const bool past_end = !s.loop && s.duration_s > 0.0f &&
                                  s.play_head_s >= s.duration_s;
            const bool faded    =  s.stopping && s.fade_out_s <= 0.0f;
            return past_end || faded;
        }), instances_.end());
}

void Engine::render(float* out, u32 frame_count, u32 channels) noexcept {
    if (out == nullptr || frame_count == 0 || channels == 0) return;
    std::fill(out, out + static_cast<usize>(frame_count) * channels, 0.0f);

    std::lock_guard<std::mutex> lg(mtx_);
    const float sr     = (desc_.sample_rate > 0)
                       ? static_cast<float>(desc_.sample_rate) : 48000.0f;
    const float inv_sr = 1.0f / sr;
    constexpr float kTwoPi = 6.28318530717958647692f;

    for (const auto& s : instances_) {
        const float gain = s.final_attenuated_volume;
        if (gain <= 1e-5f) continue;
        // Unlocked cue lookup — we already hold mtx_ (find_cue would
        // re-lock → deadlock).
        auto cit = cues_.find(s.cue_id);
        if (cit == cues_.end()) continue;
        const Cue& c = cit->second;
        if (c.kind == CueKind::Silence || c.kind == CueKind::Streamed) continue;

        for (u32 f = 0; f < frame_count; ++f) {
            const float t = s.play_head_s
                          + static_cast<float>(f) * inv_sr * s.pitch;
            if (!s.loop && s.duration_s > 0.0f && t >= s.duration_s) break;
            float v = 0.0f;
            if (c.kind == CueKind::SineWave) {
                v = std::sin(kTwoPi * c.sine_frequency_hz * t) * c.gain;
            } else if (c.kind == CueKind::Procedural && c.procedural) {
                v = c.procedural(t);
            }
            v *= gain;
            float* fr = out + static_cast<usize>(f) * channels;
            for (u32 ch = 0; ch < channels; ++ch) fr[ch] += v;
        }
    }

    // Soft clip the summed bus so overlapping cues don't hard-distort.
    const usize n = static_cast<usize>(frame_count) * channels;
    for (usize i = 0; i < n; ++i) {
        const float x = out[i];
        out[i] = (x > 1.0f) ? 1.0f : (x < -1.0f ? -1.0f : x);
    }
}

EngineStats Engine::stats() const {
    EngineStats s{};
    std::lock_guard<std::mutex> lg(mtx_);
    s.active_instances = static_cast<u32>(instances_.size());
    s.cues_registered  = static_cast<u32>(cues_.size());
    s.channels         = static_cast<u32>(channels_.size());
    s.total_played     = total_played_;
    s.total_culled     = total_culled_;
    return s;
}

std::vector<InstanceState> Engine::active_instances() const {
    std::lock_guard<std::mutex> lg(mtx_);
    return instances_;
}

}  // namespace cardinal::audio
