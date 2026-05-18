#include <cardinal/sky/sky.hpp>

#include <algorithm>
#include <cmath>

namespace cardinal::sky {

namespace {

inline cardinal::scene::Vec3 lerp_vec(const cardinal::scene::Vec3& a,
                                       const cardinal::scene::Vec3& b, float t)
{
    return cardinal::scene::Vec3{
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    };
}

constexpr float kTwoPi = 6.28318530717958647692f;

}  // namespace

Sky::Sky() {
    reset_to_defaults();
    recompute_state_();
}

void Sky::reset_to_defaults() {
    keys_ = {
        // 24h color cycle — feel free to author over this in the panel.
        // hour, zenith,                horizon,                  sun_color,                  sun_int, ambient
        { 0.00f, {0.02f, 0.02f, 0.06f}, {0.04f, 0.04f, 0.10f},  {0.55f, 0.65f, 0.95f}, 0.00f, {0.04f, 0.05f, 0.10f}},
        { 5.00f, {0.10f, 0.10f, 0.18f}, {0.40f, 0.20f, 0.10f},  {0.85f, 0.55f, 0.30f}, 0.10f, {0.10f, 0.08f, 0.10f}},
        { 7.00f, {0.30f, 0.45f, 0.78f}, {0.90f, 0.70f, 0.45f},  {1.00f, 0.85f, 0.65f}, 0.85f, {0.18f, 0.16f, 0.18f}},
        {12.00f, {0.30f, 0.55f, 0.95f}, {0.65f, 0.78f, 0.92f},  {1.00f, 0.97f, 0.92f}, 1.20f, {0.20f, 0.20f, 0.22f}},
        {17.00f, {0.30f, 0.50f, 0.85f}, {0.85f, 0.65f, 0.50f},  {1.00f, 0.80f, 0.55f}, 0.90f, {0.18f, 0.16f, 0.16f}},
        {19.00f, {0.18f, 0.16f, 0.36f}, {0.65f, 0.32f, 0.20f},  {0.90f, 0.45f, 0.25f}, 0.30f, {0.10f, 0.07f, 0.10f}},
        {21.00f, {0.05f, 0.06f, 0.16f}, {0.10f, 0.10f, 0.20f},  {0.45f, 0.50f, 0.85f}, 0.05f, {0.05f, 0.06f, 0.10f}},
        {23.99f, {0.02f, 0.02f, 0.06f}, {0.04f, 0.04f, 0.10f},  {0.55f, 0.65f, 0.95f}, 0.00f, {0.04f, 0.05f, 0.10f}},
    };
}

void Sky::sort_keys() {
    std::sort(keys_.begin(), keys_.end(),
              [](const SkyKey& a, const SkyKey& b){ return a.hour < b.hour; });
}

void Sky::set_hour(float h) {
    h = std::fmod(h, 24.0f);
    if (h < 0.0f) h += 24.0f;
    state_.hour = h;
    recompute_state_();
}

void Sky::set_day_length_seconds(float s) noexcept {
    if (s < 0.001f) s = 0.001f;
    // 24 in-game hours per `s` real seconds → time_scale (hours/sec) = 24/s
    // We store time_scale_ as "hours per real second" * 1.0f to avoid
    // re-deriving each frame.
    time_scale_ = 24.0f / s;
}

float Sky::day_length_seconds() const noexcept {
    return time_scale_ > 1e-6f ? (24.0f / time_scale_) : 0.0f;
}

void Sky::tick(float real_dt) {
    if (frozen_) return;
    state_.hour += real_dt * time_scale_;
    if (state_.hour >= 24.0f) state_.hour = std::fmod(state_.hour, 24.0f);
    if (state_.hour <  0.0f)  state_.hour += 24.0f;
    recompute_state_();
}

void Sky::recompute_state_() {
    if (keys_.empty()) return;

    // Find segment surrounding state_.hour.
    float h = state_.hour;
    SkyKey k0 = keys_.back();   // wrap left
    SkyKey k1 = keys_.front();  // wrap right
    bool found = false;
    for (usize i = 0; i + 1 < keys_.size(); ++i) {
        if (h >= keys_[i].hour && h <= keys_[i + 1].hour) {
            k0 = keys_[i]; k1 = keys_[i + 1]; found = true; break;
        }
    }
    if (!found) {
        // h is in the wrap-around segment from last key to (first key + 24h).
        k0 = keys_.back();
        k1 = keys_.front();
    }
    const float span = (k1.hour > k0.hour)
        ? (k1.hour - k0.hour)
        : (k1.hour + 24.0f - k0.hour);
    const float local = (h >= k0.hour) ? (h - k0.hour) : (h + 24.0f - k0.hour);
    const float t = std::clamp(local / std::max(1e-4f, span), 0.0f, 1.0f);

    state_.zenith        = lerp_vec(k0.zenith,    k1.zenith,    t);
    state_.horizon       = lerp_vec(k0.horizon,   k1.horizon,   t);
    state_.sun_color     = lerp_vec(k0.sun_color, k1.sun_color, t);
    state_.sun_intensity = k0.sun_intensity + (k1.sun_intensity - k0.sun_intensity) * t;
    state_.ambient       = lerp_vec(k0.ambient,   k1.ambient,   t);

    // Sun direction: at noon (12.0) the sun is overhead (-Y). At 6 AM
    // it's on the +X horizon, at 6 PM on the -X horizon. Below the
    // horizon (night) we still emit a unit vector — the renderer reads
    // sun_intensity to decide whether to apply.
    const float angle = ((state_.hour - 6.0f) / 12.0f) * 3.14159265358979f;
    cardinal::scene::Vec3 to_sun{
        std::cos(angle),
        std::sin(angle),
        0.10f                      // slight Z tilt — looks better than perfectly head-on
    };
    // Direction the light *travels* (i.e. away from sun toward the world).
    state_.sun_dir = cardinal::scene::Vec3{ -to_sun.x, -to_sun.y, -to_sun.z };
    // Normalise.
    const float l = std::sqrt(state_.sun_dir.x * state_.sun_dir.x +
                              state_.sun_dir.y * state_.sun_dir.y +
                              state_.sun_dir.z * state_.sun_dir.z);
    if (l > 1e-6f) {
        state_.sun_dir.x /= l;
        state_.sun_dir.y /= l;
        state_.sun_dir.z /= l;
    }
}

}  // namespace cardinal::sky
