#pragma once

// =============================================================================
// Cardinal — Cinematic Sequencer.
//
// A Sequence is a multi-track timeline of events that fire over time:
//
//   - AnimationTrack — drives a cardinal::anim::Player (apply curves)
//   - CameraTrack    — switches the active camera at shot boundaries
//   - AudioTrack     — schedules audio cues (cardinal::audio::Engine play_*)
//   - EventTrack     — fires a named event (string + payload) at time T;
//                      the host wires what it means
//
// Plus shared bookkeeping:
//
//   - Markers        — named time points for the editor (chapter heads, etc.)
//   - Regions        — labeled time intervals (scene blocks)
//   - Render config  — output frame range / fps / path the editor uses to
//                      produce a "render to disk" job
//
// The Sequencer is engine-side; the editor's timeline UI (panels/sequencer)
// reads + mutates this struct.
// =============================================================================

#include <cardinal/anim/anim.hpp>
#include <cardinal/audio/audio.hpp>
#include <cardinal/core/types.hpp>        // function/string/shared_ptr
#include <cardinal/core/std/utility.hpp>      // cardinal::variant, cardinal::move
#include <cardinal/core/std/containers.hpp>   // cardinal::vector

namespace cardinal::cine {

// ---------------------------------------------------------------------------
// Tracks
// ---------------------------------------------------------------------------
struct AnimationTrack {
    cardinal::string                       name;
    cardinal::shared_ptr<cardinal::anim::Player> player;
    float                             start_time_s{0.0f};
    float                             length_s   {1.0f};
};

struct CameraShot {
    cardinal::string label;
    float       time_s{0.0f};
    u32         camera_actor_id{0};
    float       blend_seconds{0.5f};   // 0 = cut
};

struct CameraTrack {
    cardinal::string             name;
    cardinal::vector<CameraShot> shots;     // sorted by time
};

struct AudioCueEvent {
    float       time_s{0.0f};
    cardinal::string cue_id;
    cardinal::audio::ChannelId channel{cardinal::audio::kChannelMusic};
    float       volume{1.0f};
    bool        is_3d{false};
    cardinal::scene::Vec3 position{0,0,0};
};

struct AudioTrack {
    cardinal::string                 name;
    cardinal::vector<AudioCueEvent>  events;     // sorted by time
};

struct EventCall {
    float       time_s{0.0f};
    cardinal::string event_name;
    cardinal::string payload;
};

struct EventTrack {
    cardinal::string             name;
    cardinal::vector<EventCall>  events;
};

using Track = cardinal::variant<AnimationTrack, CameraTrack, AudioTrack, EventTrack>;

// ---------------------------------------------------------------------------
// Markers + regions
// ---------------------------------------------------------------------------
struct Marker {
    cardinal::string label;
    float       time_s{0.0f};
    u32         color_rgba{0xFFFFFFFFu};
};

struct Region {
    cardinal::string label;
    float       start_s{0.0f};
    float       end_s  {1.0f};
    u32         color_rgba{0x80808080u};
};

// ---------------------------------------------------------------------------
// Render config
// ---------------------------------------------------------------------------
struct RenderConfig {
    cardinal::string output_path{"capture/sequence_%04u.png"};
    float       start_time_s{0.0f};
    float       end_time_s  {5.0f};
    u32         fps         {30};
    u32         width       {1920};
    u32         height      {1080};
};

// ---------------------------------------------------------------------------
// Sequence
// ---------------------------------------------------------------------------
struct Sequence {
    cardinal::string         name;
    float               duration_s{5.0f};
    cardinal::vector<Track>  tracks;
    cardinal::vector<Marker> markers;
    cardinal::vector<Region> regions;
    RenderConfig        render;

    // Editor playback state
    float    play_head_s{0.0f};
    bool     playing{false};
    bool     looping{true};
    float    speed{1.0f};
};

// ---------------------------------------------------------------------------
// Player — advances the play head, fires track effects.
//
// Audio + Event tracks are *edge-triggered*: each event fires once when the
// playhead crosses its time. Animation tracks just push their player to
// (playhead - start_time_s).
// ---------------------------------------------------------------------------
class Player {
public:
    explicit Player(Sequence* seq, cardinal::audio::Engine* audio = nullptr)
        : seq_(seq), audio_(audio) {}

    Sequence* sequence() noexcept { return seq_; }

    using EventDispatch = cardinal::function<void(const cardinal::string&, const cardinal::string&)>;
    using CameraSwitch  = cardinal::function<void(u32 actor_id, float blend_seconds)>;
    void set_event_dispatch(EventDispatch fn) { on_event_  = cardinal::move(fn); }
    void set_camera_switch(CameraSwitch fn)   { on_camera_ = cardinal::move(fn); }

    void play()  noexcept;
    void pause() noexcept;
    void stop()  noexcept;
    void seek(float t);

    void tick(float dt);

private:
    Sequence*               seq_{nullptr};
    cardinal::audio::Engine* audio_{nullptr};
    EventDispatch           on_event_;
    CameraSwitch            on_camera_;
    float                   prev_time_s_{0.0f};
};

}  // namespace cardinal::cine
