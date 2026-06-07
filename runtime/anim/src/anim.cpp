#include <cardinal/anim/anim.hpp>

#include <cardinal/core/std/algorithm.hpp>   // cardinal::lower_bound/max
#include <cardinal/core/std/cmath.hpp>       // cardinal::fmod

namespace cardinal::anim {

namespace {

float wrap_time(float t, float duration, WrapMode mode) noexcept {
    if (duration <= 1e-6f) return 0.0f;
    switch (mode) {
        case WrapMode::Clamp:
            if (t < 0.0f) return 0.0f;
            if (t > duration) return duration;
            return t;
        case WrapMode::Loop: {
            float r = cardinal::fmod(t, duration);
            if (r < 0.0f) r += duration;
            return r;
        }
        case WrapMode::PingPong: {
            const float two = 2.0f * duration;
            float r = cardinal::fmod(t, two);
            if (r < 0.0f) r += two;
            return r > duration ? (two - r) : r;
        }
    }
    return t;
}

template <class T>
T lerp(const T& a, const T& b, float u);

template <>
float lerp<float>(const float& a, const float& b, float u) { return a + (b - a) * u; }

template <>
cardinal::scene::Vec3 lerp<cardinal::scene::Vec3>(
    const cardinal::scene::Vec3& a, const cardinal::scene::Vec3& b, float u)
{
    return cardinal::scene::Vec3{
        a.x + (b.x - a.x) * u,
        a.y + (b.y - a.y) * u,
        a.z + (b.z - a.z) * u,
    };
}

template <>
cardinal::scene::Vec4 lerp<cardinal::scene::Vec4>(
    const cardinal::scene::Vec4& a, const cardinal::scene::Vec4& b, float u)
{
    return cardinal::scene::Vec4{
        a.x + (b.x - a.x) * u,
        a.y + (b.y - a.y) * u,
        a.z + (b.z - a.z) * u,
        a.w + (b.w - a.w) * u,
    };
}

template <class T>
T hermite(const T& v0, const T& m0, const T& v1, const T& m1, float u);

template <>
float hermite<float>(const float& v0, const float& m0, const float& v1,
                     const float& m1, float u)
{
    const float u2 = u * u, u3 = u2 * u;
    const float h00 =  2.0f*u3 - 3.0f*u2 + 1.0f;
    const float h10 =        u3 - 2.0f*u2 + u;
    const float h01 = -2.0f*u3 + 3.0f*u2;
    const float h11 =        u3 -        u2;
    return h00*v0 + h10*m0 + h01*v1 + h11*m1;
}

template <>
cardinal::scene::Vec3 hermite<cardinal::scene::Vec3>(
    const cardinal::scene::Vec3& v0, const cardinal::scene::Vec3& m0,
    const cardinal::scene::Vec3& v1, const cardinal::scene::Vec3& m1,
    float u)
{
    return cardinal::scene::Vec3{
        hermite<float>(v0.x, m0.x, v1.x, m1.x, u),
        hermite<float>(v0.y, m0.y, v1.y, m1.y, u),
        hermite<float>(v0.z, m0.z, v1.z, m1.z, u),
    };
}

template <>
cardinal::scene::Vec4 hermite<cardinal::scene::Vec4>(
    const cardinal::scene::Vec4& v0, const cardinal::scene::Vec4& m0,
    const cardinal::scene::Vec4& v1, const cardinal::scene::Vec4& m1,
    float u)
{
    return cardinal::scene::Vec4{
        hermite<float>(v0.x, m0.x, v1.x, m1.x, u),
        hermite<float>(v0.y, m0.y, v1.y, m1.y, u),
        hermite<float>(v0.z, m0.z, v1.z, m1.z, u),
        hermite<float>(v0.w, m0.w, v1.w, m1.w, u),
    };
}

}  // namespace

template <class T>
T Curve<T>::sample(float t) const noexcept {
    if (keys.empty()) return T{};
    if (keys.size() == 1) return keys[0].value;
    const float dur = duration();
    t = wrap_time(t, dur, wrap);
    // A non-finite time (NaN dt/speed in Player::tick, or fmod() of a
    // non-finite accumulated time in the Loop/PingPong wrap) makes BOTH
    // ordered clamp guards below false — NaN compares unordered — so
    // lower_bound() returns begin(), i1 becomes 0 and i0 = i1 - 1 wraps to
    // SIZE_MAX: an out-of-bounds keys[] read in this noexcept function.
    // Treat a non-finite time as the clamp-to-front case.
    if (t != t) return keys.front().value;
    if (t <= keys.front().time) return keys.front().value;
    if (t >= keys.back().time)  return keys.back().value;
    // Binary search for the segment.
    auto it = cardinal::lower_bound(keys.begin(), keys.end(), t,
        [](const Key<T>& k, float v){ return k.time < v; });
    const usize i1 = static_cast<usize>(it - keys.begin());
    const usize i0 = i1 - 1;
    const Key<T>& k0 = keys[i0];
    const Key<T>& k1 = keys[i1];
    const float span = cardinal::max(1e-6f, k1.time - k0.time);
    const float u    = (t - k0.time) / span;
    switch (k0.mode) {
        case InterpMode::Step:         return k0.value;
        case InterpMode::Linear:       return lerp<T>(k0.value, k1.value, u);
        case InterpMode::CubicHermite: return hermite<T>(k0.value, k0.out_tangent,
                                                          k1.value, k1.in_tangent, u);
    }
    return k0.value;
}

// Explicit instantiations — the renderer picks these up.
template struct Curve<float>;
template struct Curve<cardinal::scene::Vec3>;
template struct Curve<cardinal::scene::Vec4>;

void Clip::add_track(cardinal::unique_ptr<ITrack> tr) {
    tracks_.push_back(cardinal::move(tr));
}
float Clip::duration() const noexcept {
    float d = 0.0f;
    for (const auto& t : tracks_) d = cardinal::max(d, t->duration());
    return d;
}
void Clip::apply(float t) {
    for (auto& tr : tracks_) tr->apply(t);
}

// Out-of-line setters with NaN/Inf guards (declared inline-style in the
// header but kept here so the header doesn't depend on cmath).
// time_ feeds into the `time_ += dt * speed_` accumulator in tick;
// either a non-finite time_ (seek) or a non-finite speed_ (set_speed)
// permanently poisons the accumulator. Curve::sample's downstream
// NaN/Inf defenses (d8153cc) keep the sampled values defined, but the
// accumulator itself stays poisoned → animation frozen at the front
// value forever. Reject at the source.
void Player::seek(float t) {
    if (!cardinal::isfinite(t)) t = 0.0f;
    time_ = t;
}
void Player::set_speed(float s) {
    if (!cardinal::isfinite(s)) s = 0.0f;
    speed_ = s;
}

void Player::tick(float dt) {
    if (!playing_ || clip_ == nullptr) return;
    // Reject non-finite dt at the SOURCE — `time_ += NaN` (or *speed_==NaN)
    // poisons time_ permanently. d8153cc fixed the symptom (Curve::sample
    // clamp-to-front on NaN — kept the OOB read from happening) but time_
    // stays NaN every subsequent tick, freezing the animation at the front
    // value forever even after the upstream NaN clears. Same accumulator-
    // poison family as physics/sim/cine/particles/audio/ai. Skip the
    // bad frame and continue normally next tick.
    if (!cardinal::isfinite(dt)) return;
    time_ += dt * speed_;
    const float dur = clip_->duration();
    if (dur > 0.0f) {
        if (time_ > dur) {
            if (loop_) {
                time_ = cardinal::fmod(time_, dur);
            } else {
                time_    = dur;
                playing_ = false;
            }
        }
    }
    clip_->apply(time_);
}

}  // namespace cardinal::anim
