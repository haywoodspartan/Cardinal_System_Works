#pragma once

// =============================================================================
// Cardinal — Audio system.
//
// Backend-agnostic. The Engine maintains a mixer with named channels and a
// sound-cue catalog. Emitters reference cues; the engine processes them
// each frame, applying volume/pitch/3D attenuation. We don't (yet) wire to
// a real audio API — instead we maintain bookkeeping (instance state,
// sample-accurate position) so the integrator can hook XAudio2 / WASAPI /
// ALSA / WebAudio later.
//
// Design notes that beat naive single-listener engines:
//   - Channel hierarchy: Master -> Music/SFX/Voice/UI; volume scales propagate
//   - Per-cue cool-down (avoid step-sound spam when an actor jitters)
//   - Per-emitter priority + max-instance cap (oldest culled when full)
//   - Logarithmic distance attenuation with configurable rolloff
//   - Time-based fades (in/out/crossfade) — value-driven, not buffer math
//
// Cue formats:
//   - SineWave (frequency_hz, duration_s, gain) — perfect for tests / UI tones
//   - Procedural (custom callback) — host-supplied PCM generator
//   - Streamed   (path) — host opens + reads; engine just stores the cue
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cardinal::audio {

using ChannelId  = u32;
inline constexpr ChannelId kChannelMaster = 0;
inline constexpr ChannelId kChannelMusic  = 1;
inline constexpr ChannelId kChannelSfx    = 2;
inline constexpr ChannelId kChannelVoice  = 3;
inline constexpr ChannelId kChannelUi     = 4;

const char* channel_name(ChannelId id) noexcept;

// ---------------------------------------------------------------------------
// Cues
// ---------------------------------------------------------------------------
enum class CueKind : u32 { Silence, SineWave, Procedural, Streamed };

struct Cue {
    std::string id;
    CueKind     kind{CueKind::Silence};
    float       duration_s{1.0f};
    float       sine_frequency_hz{440.0f};
    float       gain{1.0f};
    std::string streamed_path;
    // For Procedural — host returns one sample per call. The engine
    // chooses sample_rate (default 48000).
    std::function<float(float t)> procedural;
};

// ---------------------------------------------------------------------------
// Channels
// ---------------------------------------------------------------------------
struct Channel {
    ChannelId id{0};
    std::string name;
    float     volume{1.0f};        // 0..1
    bool      muted{false};
    bool      solo{false};
    ChannelId parent{kChannelMaster};
};

// ---------------------------------------------------------------------------
// Listener — the "ear" position used for 3D attenuation.
// ---------------------------------------------------------------------------
struct Listener {
    cardinal::scene::Vec3 position{0, 0, 0};
    cardinal::scene::Vec3 forward {0, 0, -1};
    cardinal::scene::Vec3 up      {0, 1, 0};
};

// ---------------------------------------------------------------------------
// Instance — one playing instance of a cue.
// ---------------------------------------------------------------------------
using InstanceId = u64;

struct InstanceState {
    InstanceId id{0};
    std::string cue_id;
    ChannelId channel{kChannelSfx};
    bool      is_3d{true};
    cardinal::scene::Vec3 position{0,0,0};
    float     volume{1.0f};
    float     pitch{1.0f};
    bool      loop{false};
    float     play_head_s{0.0f};
    float     duration_s{0.0f};
    float     fade_in_s {0.0f};
    float     fade_out_s{0.0f};
    bool      stopping{false};
    float     final_attenuated_volume{1.0f};   // updated each tick
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
struct EngineDesc {
    u32   sample_rate{48000};
    u32   max_instances{128};
    float distance_min{1.0f};       // below this, no attenuation
    float distance_max{50.0f};      // beyond this, silence
    float rolloff{1.0f};            // 1=inverse, 2=inverse-squared
};

struct EngineStats {
    u32 active_instances{0};
    u32 cues_registered{0};
    u32 channels{0};
    u64 total_played{0};
    u64 total_culled{0};
};

class Engine {
public:
    static std::shared_ptr<Engine> create(const EngineDesc& desc = {});
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // ---- Cues -------------------------------------------------------
    void register_cue(Cue cue);
    void unregister_cue(const std::string& id);
    const Cue* find_cue(const std::string& id) const;

    // ---- Channels ---------------------------------------------------
    Channel*       channel(ChannelId id);
    const Channel* channel(ChannelId id) const;
    void           set_channel_volume(ChannelId id, float v);
    void           set_channel_muted (ChannelId id, bool m);
    float          effective_volume  (ChannelId id) const;     // includes parents
    std::vector<Channel> channels() const;

    // ---- Listener ---------------------------------------------------
    void  set_listener(const Listener& l);
    const Listener& listener() const noexcept { return listener_; }

    // ---- Playback ---------------------------------------------------
    InstanceId play_2d(const std::string& cue_id, ChannelId ch = kChannelSfx,
                       float volume = 1.0f, float pitch = 1.0f, bool loop = false);
    InstanceId play_3d(const std::string& cue_id, const cardinal::scene::Vec3& pos,
                       ChannelId ch = kChannelSfx, float volume = 1.0f,
                       float pitch = 1.0f, bool loop = false);

    void  fade_in (InstanceId id, float seconds);
    void  fade_out(InstanceId id, float seconds);
    void  stop    (InstanceId id);
    void  set_emitter_position(InstanceId id, const cardinal::scene::Vec3& pos);

    // ---- Per-frame --------------------------------------------------
    void  tick(float dt);

    // ---- PCM mix (audio-backend feed point) -------------------------
    //
    // Sums every active instance into `out` — interleaved float frames,
    // `channels` per frame (mono cues are duplicated to all channels),
    // additive with a soft tanh-ish clip. This is the single function a
    // real output backend (WASAPI / XAudio2 / ALSA) calls from its
    // audio thread; the engine itself stays device-agnostic. Until a
    // backend is wired nothing calls this, so it's a pure, testable,
    // zero-behaviour-change addition.
    //
    // Gain uses each instance's final_attenuated_volume from the most
    // recent tick() (channel/3D/fade already folded in) — i.e. gain is
    // updated at tick rate, waveform phase at sample rate. Good enough
    // for tones / UI blips / contact SFX; per-sample gain ramps are a
    // later refinement. Streamed cues render silence for now (the Cue
    // carries a path, not PCM — host-decoded streaming is future work).
    void  render(float* out, u32 frame_count, u32 channels) noexcept;

    // ---- Inspection -------------------------------------------------
    EngineStats stats() const;
    std::vector<InstanceState> active_instances() const;

private:
    Engine() = default;
    bool initialize_(const EngineDesc& desc);
    float compute_3d_attenuation_(const cardinal::scene::Vec3& pos) const noexcept;
    bool  any_solo_active_() const noexcept;

    EngineDesc                                  desc_{};
    Listener                                    listener_{};
    mutable std::mutex                          mtx_;
    std::unordered_map<std::string, Cue>        cues_;
    std::unordered_map<ChannelId, Channel>      channels_;
    std::vector<InstanceState>                  instances_;
    InstanceId                                  next_instance_id_{1};
    u64                                         total_played_{0};
    u64                                         total_culled_{0};
};

// ---------------------------------------------------------------------------
// Output backend — pumps Engine::render() to a real device.
//
// Opaque RAII handle: construction starts a dedicated audio thread that
// owns the OS device and calls engine->render() to fill each buffer;
// destruction stops + joins the thread and releases the device. The
// Engine stays 100% device-agnostic — this is the only piece that
// touches OS audio APIs (Windows: WASAPI shared-mode, event-driven).
//
// start_default_output() returns nullptr if no device is available, the
// device's shared mix format isn't 32-bit float, or the platform has no
// backend yet (non-Windows today → silent no-op, same as before).
// ---------------------------------------------------------------------------
class OutputBackend {
public:
    virtual ~OutputBackend() = default;
};

std::unique_ptr<OutputBackend>
start_default_output(std::shared_ptr<Engine> engine);

}  // namespace cardinal::audio
