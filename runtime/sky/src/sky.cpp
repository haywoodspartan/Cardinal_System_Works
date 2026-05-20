#include <cardinal/sky/sky.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::sort/clamp/max
#include <cardinal/core/cmath.hpp>       // cardinal::fmod/cos/sin/sqrt

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
    cardinal::sort(keys_.begin(), keys_.end(),
              [](const SkyKey& a, const SkyKey& b){ return a.hour < b.hour; });
}

void Sky::set_hour(float h) {
    // Non-finite h: fmod(NaN, 24) = NaN, `NaN < 0.0f` is false → wrap
    // skipped → state_.hour = NaN → recompute_state_ drives every
    // derived field to NaN (cardinal::clamp passes NaN through; lerp
    // of NaN is NaN). Reject up-front and treat as noon (the
    // constructor default), so the caller's bad value doesn't poison
    // a previously-good sky state.
    if (!cardinal::isfinite(h)) h = 12.0f;
    h = cardinal::fmod(h, 24.0f);
    if (h < 0.0f) h += 24.0f;
    state_.hour = h;
    recompute_state_();
}

void Sky::set_day_length_seconds(float s) noexcept {
    // Non-finite s: same SECOND-INGRESS pattern as sim::set_time_scale
    // (70a9324). The original `s < 0.001f` clamp is NaN-blind and
    // +Inf-blind. NaN s → `time_scale_ = 24.0f / NaN = NaN`. The
    // 6ac4418 Sky::tick guard rejects non-finite real_dt at the
    // ingress, but if time_scale_ itself is NaN every state_.hour
    // update for a FINITE real_dt still produces NaN (real_dt *
    // NaN = NaN). +Inf s → time_scale_ = 0, day frozen — defined but
    // useless. cardinal::isfinite catches both; preserve the existing
    // min clamp.
    if (!cardinal::isfinite(s) || s < 0.001f) s = 0.001f;
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
    // Reject non-finite real_dt — `state_.hour += NaN * time_scale_`
    // poisons state_.hour. The 24-wrap guards (`>= 24.0f`, `< 0.0f`) are
    // both false for NaN, and recompute_state_() then drives EVERY
    // derived field (zenith / horizon / sun_color / sun_intensity /
    // sun_dir) to NaN via cardinal::clamp (NaN-passthrough) + lerp_vec.
    // The sky goes black/undefined and stays that way forever (hour is
    // poisoned). Same accumulator-poison family as physics/sim/cine/
    // particles/audio/anim/ai. Drop the bad frame and continue normally.
    if (!cardinal::isfinite(real_dt)) return;
    state_.hour += real_dt * time_scale_;
    if (state_.hour >= 24.0f) state_.hour = cardinal::fmod(state_.hour, 24.0f);
    if (state_.hour <  0.0f)  state_.hour += 24.0f;
    recompute_state_();
}

void Sky::recompute_state_() {
    if (keys_.empty()) return;

    // SkyKey fields (hour / zenith / horizon / sun_color / sun_intensity
    // / ambient) are PUBLIC direct fields mutable via Sky::keys() — same
    // desc-direct-field NaN sink class as particles::EmitterDesc
    // (35f8da7). Any non-finite component poisons state_ via lerp_vec
    // (NaN-passthrough), then renderer reads NaN colors / NaN
    // sun_intensity / NaN sun_dir. NaN hour additionally defeats the
    // segment search (`h >= NaN` / `h <= NaN` both unordered-false),
    // forcing the wrap-around branch and a NaN `span` / NaN `t`. The
    // safe_key helper coerces every non-finite component to 0 BEFORE
    // the segment search and lerp, keeping state_ finite-by-
    // construction. User's stored keys_ are preserved.
    auto fz = [](float v) noexcept {
        return cardinal::isfinite(v) ? v : 0.0f;
    };
    auto safe_key = [&](const SkyKey& k) noexcept -> SkyKey {
        SkyKey s = k;
        s.hour          = fz(k.hour);
        s.zenith        = { fz(k.zenith.x),    fz(k.zenith.y),    fz(k.zenith.z) };
        s.horizon       = { fz(k.horizon.x),   fz(k.horizon.y),   fz(k.horizon.z) };
        s.sun_color     = { fz(k.sun_color.x), fz(k.sun_color.y), fz(k.sun_color.z) };
        s.sun_intensity = fz(k.sun_intensity);
        s.ambient       = { fz(k.ambient.x),   fz(k.ambient.y),   fz(k.ambient.z) };
        return s;
    };

    // Find segment surrounding state_.hour. Compare against SANITIZED
    // hours so a NaN-hour key doesn't defeat the search.
    float h = state_.hour;
    SkyKey k0 = safe_key(keys_.back());   // wrap left
    SkyKey k1 = safe_key(keys_.front());  // wrap right
    bool found = false;
    for (usize i = 0; i + 1 < keys_.size(); ++i) {
        const float h_i  = fz(keys_[i].hour);
        const float h_i1 = fz(keys_[i + 1].hour);
        if (h >= h_i && h <= h_i1) {
            k0 = safe_key(keys_[i]); k1 = safe_key(keys_[i + 1]);
            found = true; break;
        }
    }
    if (!found) {
        // h is in the wrap-around segment from last key to (first key + 24h).
        k0 = safe_key(keys_.back());
        k1 = safe_key(keys_.front());
    }
    const float span = (k1.hour > k0.hour)
        ? (k1.hour - k0.hour)
        : (k1.hour + 24.0f - k0.hour);
    const float local = (h >= k0.hour) ? (h - k0.hour) : (h + 24.0f - k0.hour);
    const float t = cardinal::clamp(local / cardinal::max(1e-4f, span), 0.0f, 1.0f);

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
        cardinal::cos(angle),
        cardinal::sin(angle),
        0.10f                      // slight Z tilt — looks better than perfectly head-on
    };
    // Direction the light *travels* (i.e. away from sun toward the world).
    state_.sun_dir = cardinal::scene::Vec3{ -to_sun.x, -to_sun.y, -to_sun.z };
    // Normalise.
    const float l = cardinal::sqrt(state_.sun_dir.x * state_.sun_dir.x +
                              state_.sun_dir.y * state_.sun_dir.y +
                              state_.sun_dir.z * state_.sun_dir.z);
    if (l > 1e-6f) {
        state_.sun_dir.x /= l;
        state_.sun_dir.y /= l;
        state_.sun_dir.z /= l;
    }
}

}  // namespace cardinal::sky
