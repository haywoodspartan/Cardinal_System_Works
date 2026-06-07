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
// Tracks are bound to a write-target via cardinal::function — the host sets up
// the binding when the clip is created. Generic enough to drive transforms,
// material parameters, post-process, audio volume, anything float-or-vec.
// =============================================================================

#include <cardinal/core/types.hpp>        // function/string/unique_ptr/shared_ptr
#include <cardinal/core/std/algorithm.hpp>    // cardinal::lower_bound
#include <cardinal/core/std/cmath.hpp>        // cardinal::isfinite
#include <cardinal/core/std/containers.hpp>   // cardinal::vector
#include <cardinal/core/std/utility.hpp>      // cardinal::move
#include <cardinal/scene/math.hpp>

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
    cardinal::vector<Key<T>> keys;
    WrapMode            wrap{WrapMode::Clamp};

    void add_key(float t, const T& v, InterpMode m = InterpMode::Linear) {
        Key<T> k{}; k.time = t; k.value = v; k.mode = m;
        // Insert sorted by time. NaN-safe strict-weak-ordering:
        // `a.time < b.time` is NaN-blind both ways → (NaN, x) is
        // treated as equivalent while (x, y) with x<y orders
        // strictly → transitivity-of-equivalence broken → std::sort
        // / std::lower_bound on the resulting non-partitioned range
        // is UB (same SWO class as sky 4ff85a8 / level 4b08e0c).
        // Realistic ingress: curve_editor double-click adds a key
        // at t = t_of_x(MousePos.x) with no max(0, ...) clamp; a
        // deserializer with sscanf("%f", ...) accepts "nan"/"inf"
        // verbatim. Promote NaN to "greater than all finites":
        // every NaN-time key sinks to the tail as a single
        // equivalence class, the vector stays partitioned, and
        // sample()'s downstream lower_bound (anim.cpp:121) keeps
        // the correct segment behaviour with finite t.
        auto it = cardinal::lower_bound(keys.begin(), keys.end(), k,
            [](const Key<T>& a, const Key<T>& b){
                const bool af = cardinal::isfinite(a.time);
                const bool bf = cardinal::isfinite(b.time);
                if (af && bf) return a.time < b.time;
                return af && !bf;     // finite < NaN; NaN ~ NaN
            });
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
    virtual const cardinal::string& name() const noexcept = 0;
};

template <class T>
class Track final : public ITrack {
public:
    Track(cardinal::string n, cardinal::function<void(const T&)> writer)
        : name_(cardinal::move(n)), writer_(cardinal::move(writer)) {}

    Curve<T>& curve() noexcept { return curve_; }
    const Curve<T>& curve() const noexcept { return curve_; }

    void apply(float t) override {
        if (curve_.empty() || !writer_) return;
        writer_(curve_.sample(t));
    }
    float duration() const noexcept override { return curve_.duration(); }
    const cardinal::string& name() const noexcept override { return name_; }

private:
    cardinal::string                       name_;
    cardinal::function<void(const T&)>     writer_;
    Curve<T>                          curve_;
};

// ---------------------------------------------------------------------------
// Clip — N tracks. Duration = max(track.duration()).
// ---------------------------------------------------------------------------
class Clip {
public:
    explicit Clip(cardinal::string name) : name_(cardinal::move(name)) {}

    void  add_track(cardinal::unique_ptr<ITrack> tr);
    float duration() const noexcept;
    const cardinal::string& name() const noexcept { return name_; }
    const cardinal::vector<cardinal::unique_ptr<ITrack>>& tracks() const noexcept { return tracks_; }

    void apply(float t);

private:
    cardinal::string                                name_;
    cardinal::vector<cardinal::unique_ptr<ITrack>>       tracks_;
};

// ---------------------------------------------------------------------------
// Player — drives a Clip. play()/pause()/seek()/set_loop().
// ---------------------------------------------------------------------------
class Player {
public:
    explicit Player(cardinal::shared_ptr<Clip> clip = nullptr) : clip_(cardinal::move(clip)) {}

    void  set_clip(cardinal::shared_ptr<Clip> c) { clip_ = cardinal::move(c); time_ = 0.0f; }
    Clip* clip() noexcept { return clip_.get(); }

    void  play()  { playing_ = true; }
    void  pause() { playing_ = false; }
    void  stop()  { playing_ = false; time_ = 0.0f; }
    // seek / set_speed are NaN-guarded — defined out-of-line in anim.cpp
    // to keep this header free of the cmath dependency. A non-finite
    // time_ or speed_ would poison the time_ accumulator (`time_ += dt
    // * speed_` in tick) — Curve::sample's d8153cc downstream defense
    // handles the bad time_ at sample time, but the accumulator stays
    // poisoned, so reject at the SETTER.
    void  seek(float t);
    void  set_loop(bool l)  { loop_ = l; }
    void  set_speed(float s);

    bool  playing() const noexcept { return playing_; }
    bool  looping() const noexcept { return loop_; }
    float time   () const noexcept { return time_; }
    float speed  () const noexcept { return speed_; }
    float duration() const noexcept { return clip_ ? clip_->duration() : 0.0f; }

    void tick(float dt);

private:
    cardinal::shared_ptr<Clip> clip_;
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
