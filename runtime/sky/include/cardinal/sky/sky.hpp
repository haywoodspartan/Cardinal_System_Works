#pragma once

// =============================================================================
// Cardinal — Sky / Time-of-Day system.
//
// One Sky owns:
//   - a TimeOfDay clock (0..24h, advanced each frame by `time_scale * dt`)
//   - a Skybox descriptor (zenith + horizon colors per phase, sun colors)
//   - derived per-frame outputs: sun direction (Vec3 unit), sun color,
//     ambient color, horizon/zenith colors used by the renderer
//
// Phases are key-framed across 24h:
//     00:00 = night         (deep blue/purple, no sun)
//     05:30 = dawn          (warm orange horizon, low sun)
//     09:00 = morning
//     12:00 = midday        (full daylight)
//     17:00 = late afternoon
//     19:30 = dusk          (warm red/purple)
//     22:00 = early night
//
// time_scale: 1.0 = real-time (24h cycle takes 24 wall-clock hours);
//             1440.0 = 1 minute = 24h game day; etc.
// Editor convenience: a `freeze_time` boolean.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <vector>

namespace cardinal::sky {

struct SkyKey {
    float       hour    {0.0f};        // 0..24
    cardinal::scene::Vec3 zenith   {0.05f, 0.06f, 0.10f};
    cardinal::scene::Vec3 horizon  {0.10f, 0.12f, 0.18f};
    cardinal::scene::Vec3 sun_color{1.0f, 1.0f, 1.0f};
    float       sun_intensity{1.0f};
    cardinal::scene::Vec3 ambient  {0.05f, 0.06f, 0.08f};
};

struct SkyState {
    float hour{12.0f};                 // current time-of-day, 0..24
    cardinal::scene::Vec3 zenith    {0.30f, 0.50f, 0.85f};
    cardinal::scene::Vec3 horizon   {0.65f, 0.78f, 0.92f};
    cardinal::scene::Vec3 sun_dir   {0.0f, -1.0f, 0.0f};   // pointing AT scene
    cardinal::scene::Vec3 sun_color {1.0f, 1.0f, 1.0f};
    float                 sun_intensity{1.0f};
    cardinal::scene::Vec3 ambient   {0.1f, 0.1f, 0.12f};
};

class Sky {
public:
    Sky();

    // ---- Clock ---------------------------------------------------
    float       hour()        const noexcept { return state_.hour; }
    void        set_hour(float h);
    void        set_time_scale(float s) noexcept { time_scale_ = std::max(0.0f, s); }
    float       time_scale()  const noexcept { return time_scale_; }
    void        set_frozen(bool f) noexcept { frozen_ = f; }
    bool        frozen()      const noexcept { return frozen_; }

    // Game-day length in real seconds — convenience that converts to
    // time_scale. day_length=60s → 1 minute = 24 in-game hours.
    void        set_day_length_seconds(float s) noexcept;
    float       day_length_seconds() const noexcept;

    // ---- Per-frame ------------------------------------------------
    void        tick(float real_dt);

    // ---- Read-only state ----------------------------------------
    const SkyState&   state() const noexcept { return state_; }

    // ---- Keys -----------------------------------------------------
    // Direct access for editing in the panel.
    std::vector<SkyKey>&       keys()       noexcept { return keys_; }
    const std::vector<SkyKey>& keys() const noexcept { return keys_; }
    void                       sort_keys();
    void                       reset_to_defaults();

private:
    void recompute_state_();

    std::vector<SkyKey> keys_;
    SkyState            state_{};
    float               time_scale_{120.0f};   // default: 12 minutes = 24h day
    bool                frozen_{false};
};

}  // namespace cardinal::sky
