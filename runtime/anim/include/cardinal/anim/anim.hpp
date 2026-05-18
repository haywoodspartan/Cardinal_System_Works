#pragma once

// =============================================================================
// Cardinal — Keyframe Animation system.
//
// Pure-CPU curve sampling. The renderer asks "what's the value of this
// curve at time t?" and gets a typed answer — float / Vec3 / Vec4 / Quat.
//
//   Curve<T>      - sorted vector of (time, value, in/out tangent, mode)
//   Track         - typed binding: a Curve<T> + a "what to write into" path
//   Clip          - collection of Tracks bound to one or more actors
//   Player        - drives a Clip with play/pause/seek/loop semantics
//
// Interpolation modes:
//   Step          - hold previous value
//   Linear        - lerp between adjacent keys
//   CubicHermite  - cubic Hermite using user-supplied tangents (defaults are
//                   centripetal Catmull-Rom for nice editor-default curves)
//
// Tracks are bound to a write-target via std::function — the host sets up
// the binding when the clip is created. Generic enough to drive transforms,
// material parameters, post-process, audio volume, anything float-or-vec.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cardinal::anim {

enum class InterpMode : u32 { Step, Linear, CubicHermite };
enum class WrapMode   : u32 { Clamp, Loop, PingPong };

template <class T>
struct Key {
    float time{0.0f};
    T     value{};
    T     in_tangent{};
    T     out_tangent{};
    InterpMode mode{InterpMode::Linear};
};

template <class T>
struct Curve {
    std::vector<Key<T>> keys;
    WrapMode            wrap{WrapMode::Clamp};

    void add_key(float t, const T& v, InterpMode m = InterpMode::Linear) {
        Key<T> k{}; k.time = t; k.value = v; k.mode = m;
        // Insert sorted by time.
        auto it = std::lower_bound(keys.begin(), keys.end(), k,
            [](const Key<T>& a, const Key<T>& b){ return a.time < b.time; });
        keys.insert(it, k);
    }

    void clear() { keys.clear(); }
    bool empty() const noexcept { return keys.empty(); }
    float duration() const noexcept {
        return keys.empty() ? 0.0f : keys.back().time;
    }

    T sample(float t) const noexcept;
};

// ---------------------------------------------------------------------------
// Track — one Curve<T> plus a write-back binding that pushes the sampled
// value to its destination. The binding is opaque — could be writing into
// an actor's transform, a material parameter, a sequencer marker, etc.
// ---------------------------------------------------------------------------
struct ITrack {
    virtual ~ITrack() = default;
    virtual void  apply(float t)         = 0;
    virtual float duration() const noexcept = 0;
    virtual const std::string& name() const noexcept = 0;
};

template <class T>
class Track final : public ITrack {
public:
    Track(std::string n, std::function<void(const T&)> writer)
        : name_(std::move(n)), writer_(std::move(writer)) {}

    Curve<T>& curve() noexcept { return curve_; }
    const Curve<T>& curve() const noexcept { return curve_; }

    void apply(float t) override {
        if (curve_.empty() || !writer_) return;
        writer_(curve_.sample(t));
    }
    float duration() const noexcept override { return curve_.duration(); }
    const std::string& name() const noexcept override { return name_; }

private:
    std::string                       name_;
    std::function<void(const T&)>     writer_;
    Curve<T>                          curve_;
};

// ---------------------------------------------------------------------------
// Clip — N tracks. Duration = max(track.duration()).
// ---------------------------------------------------------------------------
class Clip {
public:
    explicit Clip(std::string name) : name_(std::move(name)) {}

    void  add_track(std::unique_ptr<ITrack> tr);
    float duration() const noexcept;
    const std::string& name() const noexcept { return name_; }
    const std::vector<std::unique_ptr<ITrack>>& tracks() const noexcept { return tracks_; }

    void apply(float t);

private:
    std::string                                name_;
    std::vector<std::unique_ptr<ITrack>>       tracks_;
};

// ---------------------------------------------------------------------------
// Player — drives a Clip. play()/pause()/seek()/set_loop().
// ---------------------------------------------------------------------------
class Player {
public:
    explicit Player(std::shared_ptr<Clip> clip = nullptr) : clip_(std::move(clip)) {}

    void  set_clip(std::shared_ptr<Clip> c) { clip_ = std::move(c); time_ = 0.0f; }
    Clip* clip() noexcept { return clip_.get(); }

    void  play()  { playing_ = true; }
    void  pause() { playing_ = false; }
    void  stop()  { playing_ = false; time_ = 0.0f; }
    void  seek(float t) { time_ = t; }
    void  set_loop(bool l)  { loop_ = l; }
    void  set_speed(float s){ speed_ = s; }

    bool  playing() const noexcept { return playing_; }
    bool  looping() const noexcept { return loop_; }
    float time   () const noexcept { return time_; }
    float speed  () const noexcept { return speed_; }
    float duration() const noexcept { return clip_ ? clip_->duration() : 0.0f; }

    void tick(float dt);

private:
    std::shared_ptr<Clip> clip_;
    float                 time_   {0.0f};
    float                 speed_  {1.0f};
    bool                  playing_{false};
    bool                  loop_   {false};
};

// ---------------------------------------------------------------------------
// Sample helpers — explicit instantiations for the common types live in
// the cpp; users can also instantiate Curve<T> for their own POD types.
// ---------------------------------------------------------------------------

}  // namespace cardinal::anim
