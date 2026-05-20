// =============================================================================
// Cardinal — deterministic audio-engine regression suite.
//
// cardinal::audio::Engine is the device-agnostic core (the WASAPI
// OutputBackend is the only OS-bound piece and is out of scope). Engine
// is pure bookkeeping + DSP and fully deterministic, so this suite pins
// the contract a refactor must not silently break:
//
//   * default channel tree + effective_volume parent propagation /
//     mute / clamp;
//   * cue register / find / unregister + stats;
//   * play_2d / play_3d handles, control ops (stop/fade), the
//     max_instances oldest-cull;
//   * tick — play-head advance by dt*pitch, non-loop expiry vs loop
//     persistence, volume/pitch clamps, final_attenuated_volume;
//   * 3D distance attenuation at the min/mid/max thresholds, and that
//     a 2D sound is NOT distance-attenuated;
//   * render — guards, Silence→zero, SineWave structure, soft-clip,
//     mono→all-channels duplication.
//
// It also locks a play_2d fix: the returned handle now equals the
// stored instance id (control ops used to no-op on 2D sounds) and a
// 2D sound is is_3d=false (it was 3D-at-origin, listener-attenuated).
//
// Zero deps (same harness as the other suites). Exit 0 = all pass.
// =============================================================================

#include <cardinal/audio/audio.hpp>
#include <cardinal/core/log.hpp>

#include <limits>

namespace {

namespace au = cardinal::audio;
using Vec3 = cardinal::scene::Vec3;
using cardinal::u32;
using cardinal::u64;
using cardinal::usize;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("audiotest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float eps = 1e-5f) {
    const float d = a > b ? a - b : b - a;
    return d <= eps;
}

au::Cue sine_cue(const char* id, float dur, float hz = 440.0f, float gain = 1.0f) {
    au::Cue c;
    c.id = id;
    c.kind = au::CueKind::SineWave;
    c.duration_s = dur;
    c.sine_frequency_hz = hz;
    c.gain = gain;
    return c;
}

// ---- create + channel tree + effective_volume ---------------------
void test_channels() {
    auto e = au::Engine::create();
    CHECK(e != nullptr);
    if (!e) return;

    const auto s = e->stats();
    CHECK(s.channels == 5u && s.cues_registered == 0u);
    CHECK(s.active_instances == 0u && s.total_played == 0u && s.total_culled == 0u);

    CHECK(e->channel(au::kChannelMaster) != nullptr);
    CHECK(e->channel(999u) == nullptr);
    CHECK(ap(e->channel(au::kChannelMaster)->volume, 1.0f));
    CHECK(ap(e->channel(au::kChannelMusic)->volume,  0.8f));
    CHECK(cardinal::string(au::channel_name(au::kChannelSfx)) == "SFX");
    CHECK(cardinal::string(au::channel_name(999u)) == "Channel");
    CHECK(e->channels().size() == 5u);

    // Parent propagation: child × ... × master.
    CHECK(ap(e->effective_volume(au::kChannelMaster), 1.0f));
    CHECK(ap(e->effective_volume(au::kChannelMusic),  0.8f));   // 0.8 × 1.0
    CHECK(ap(e->effective_volume(au::kChannelSfx),    1.0f));

    e->set_channel_volume(au::kChannelMaster, 0.5f);
    CHECK(ap(e->effective_volume(au::kChannelSfx),   0.5f));    // 1.0 × 0.5
    CHECK(ap(e->effective_volume(au::kChannelMusic), 0.4f));    // 0.8 × 0.5

    // Clamp to [0,1].
    e->set_channel_volume(au::kChannelSfx, 4.0f);
    CHECK(ap(e->channel(au::kChannelSfx)->volume, 1.0f));
    e->set_channel_volume(au::kChannelSfx, -2.0f);
    CHECK(ap(e->channel(au::kChannelSfx)->volume, 0.0f));

    // Mute on master kills everything beneath it.
    e->set_channel_volume(au::kChannelMaster, 1.0f);
    e->set_channel_volume(au::kChannelSfx, 1.0f);
    e->set_channel_muted(au::kChannelMaster, true);
    CHECK(ap(e->effective_volume(au::kChannelSfx),   0.0f));
    CHECK(ap(e->effective_volume(au::kChannelMusic), 0.0f));
    e->set_channel_muted(au::kChannelMaster, false);
    // Mute a leaf channel — siblings unaffected.
    e->set_channel_muted(au::kChannelSfx, true);
    CHECK(ap(e->effective_volume(au::kChannelSfx),   0.0f));
    CHECK(ap(e->effective_volume(au::kChannelMusic), 0.8f));
}

// ---- cue registry -------------------------------------------------
void test_cues() {
    auto e = au::Engine::create();
    CHECK(e->find_cue("sine") == nullptr);
    e->register_cue(sine_cue("sine", 1.0f));
    const au::Cue* c = e->find_cue("sine");
    CHECK(c != nullptr);
    if (c) CHECK(c->kind == au::CueKind::SineWave && ap(c->duration_s, 1.0f));
    CHECK(e->stats().cues_registered == 1u);
    e->register_cue(sine_cue("blip", 0.2f));
    CHECK(e->stats().cues_registered == 2u);
    e->unregister_cue("sine");
    CHECK(e->find_cue("sine") == nullptr);
    CHECK(e->stats().cues_registered == 1u);
}

// ---- play / control handles + the play_2d fix ---------------------
void test_play_control() {
    auto e = au::Engine::create();
    // Unknown cue → 0, no instance.
    CHECK(e->play_2d("nope") == 0u);
    CHECK(e->stats().total_played == 0u);

    e->register_cue(sine_cue("sine", 1.0f));
    const au::InstanceId id = e->play_2d("sine", au::kChannelSfx, 1.0f, 1.0f, false);
    CHECK(id != 0u);
    CHECK((id >> 63) == 0u);                       // no bogus 2D-marker bit
    auto act = e->active_instances();
    CHECK(act.size() == 1u);
    if (!act.empty()) {
        CHECK(act[0].id == id);                    // handle round-trips
        CHECK(act[0].is_3d == false);              // 2D ⇒ no 3D attenuation
        CHECK(act[0].cue_id == "sine" && act[0].channel == au::kChannelSfx);
        CHECK(ap(act[0].duration_s, 1.0f));
    }
    CHECK(e->stats().total_played == 1u && e->stats().active_instances == 1u);

    // stop() on the 2D handle actually removes it (was a silent no-op).
    e->stop(id);
    CHECK(e->active_instances().empty());

    // fade_out() on a 2D handle marks the instance stopping.
    const au::InstanceId id2 = e->play_2d("sine", au::kChannelSfx);
    e->fade_out(id2, 0.5f);
    auto a2 = e->active_instances();
    CHECK(a2.size() == 1u && a2[0].stopping == true && ap(a2[0].fade_out_s, 0.5f));

    // play_3d marks is_3d and round-trips its id too.
    const au::InstanceId id3 = e->play_3d("sine", {1,2,3}, au::kChannelSfx);
    bool found3 = false;
    for (const auto& s : e->active_instances())
        if (s.id == id3) { found3 = true; CHECK(s.is_3d == true); }
    CHECK(found3);
}

// ---- max_instances oldest-cull ------------------------------------
void test_instance_cull() {
    au::EngineDesc d{}; d.max_instances = 2;
    auto e = au::Engine::create(d);
    e->register_cue(sine_cue("sine", 5.0f));
    e->play_2d("sine"); e->play_2d("sine"); e->play_2d("sine");
    const auto s = e->stats();
    CHECK(s.active_instances == 2u);
    CHECK(s.total_played == 3u);
    CHECK(s.total_culled == 1u);
}

// ---- tick: play-head, expiry, loop, clamps ------------------------
void test_tick() {
    auto e = au::Engine::create();
    e->register_cue(sine_cue("sine", 1.0f));

    const au::InstanceId id = e->play_2d("sine", au::kChannelSfx, 1.0f, 1.0f, false);
    e->tick(0.25f);
    auto a = e->active_instances();
    CHECK(a.size() == 1u);
    if (!a.empty()) {
        CHECK(a[0].id == id);                      // handle round-trips
        CHECK(ap(a[0].play_head_s, 0.25f));        // dt × pitch(1)
        CHECK(ap(a[0].final_attenuated_volume, 1.0f));  // 2D, full chain
    }
    e->tick(1.0f);                                  // 1.25 ≥ 1.0 ⇒ expired
    CHECK(e->active_instances().empty());

    // pitch scales the play head; volume/pitch clamp at the play call.
    const au::InstanceId fast = e->play_2d("sine", au::kChannelSfx, 10.0f, 2.0f, false);
    (void)fast;
    auto pf = e->active_instances();
    CHECK(pf.size() == 1u);
    if (!pf.empty()) {
        CHECK(ap(pf[0].volume, 4.0f));             // clamp 0..4
        CHECK(ap(pf[0].pitch, 2.0f));
    }
    e->tick(0.5f);
    pf = e->active_instances();
    if (!pf.empty()) CHECK(ap(pf[0].play_head_s, 1.0f));   // 0.5 × 2.0

    // Looping instance is never expired by tick.
    auto e2 = au::Engine::create();
    e2->register_cue(sine_cue("sine", 0.1f));
    e2->play_2d("sine", au::kChannelSfx, 1.0f, 1.0f, /*loop=*/true);
    e2->tick(1.0f); e2->tick(1.0f);
    CHECK(e2->active_instances().size() == 1u);
}

// ---- 3D distance attenuation (and 2D bypass) ----------------------
void test_attenuation() {
    au::EngineDesc d{};                              // min 1, max 50, rolloff 1
    auto e = au::Engine::create(d);
    e->register_cue(sine_cue("sine", 10.0f));
    au::Listener L; L.position = {0,0,0};
    e->set_listener(L);

    e->play_3d("sine", {0,0,0}, au::kChannelSfx);    // d=0 ≤ min ⇒ 1.0
    e->tick(0.01f);
    CHECK(ap(e->active_instances()[0].final_attenuated_volume, 1.0f, 1e-4f));

    auto e2 = au::Engine::create(d);
    e2->register_cue(sine_cue("sine", 10.0f));
    e2->set_listener(L);
    e2->play_3d("sine", {100,0,0}, au::kChannelSfx); // d=100 ≥ max ⇒ 0.0
    e2->tick(0.01f);
    CHECK(ap(e2->active_instances()[0].final_attenuated_volume, 0.0f, 1e-4f));

    auto e3 = au::Engine::create(d);
    e3->register_cue(sine_cue("sine", 10.0f));
    e3->set_listener(L);
    e3->play_3d("sine", {25.5f,0,0}, au::kChannelSfx);  // t=0.5, pow(.5,1)=.5
    e3->tick(0.01f);
    CHECK(ap(e3->active_instances()[0].final_attenuated_volume, 0.5f, 1e-3f));

    // A 2D sound ignores listener distance entirely (the fix).
    auto e4 = au::Engine::create(d);
    e4->register_cue(sine_cue("sine", 10.0f));
    e4->set_listener(L);
    e4->play_2d("sine", au::kChannelSfx);
    e4->set_emitter_position(0, {999,0,0});          // bogus id → no effect
    e4->tick(0.01f);
    CHECK(ap(e4->active_instances()[0].final_attenuated_volume, 1.0f, 1e-4f));
}

// ---- render: guards, silence, sine structure, clip ----------------
void test_render() {
    auto e = au::Engine::create();
    float buf[128];

    // Guards — no crash / no-op.
    e->render(nullptr, 16, 2);
    e->render(buf, 0, 2);
    e->render(buf, 16, 0);

    // Silence cue → all zeros.
    au::Cue sil; sil.id = "sil"; sil.kind = au::CueKind::Silence;
    e->register_cue(sil);
    e->play_2d("sil");
    for (float& x : buf) x = 7.0f;
    e->render(buf, 32, 2);
    bool all_zero = true;
    for (u32 i = 0; i < 64u; ++i) if (buf[i] != 0.0f) all_zero = false;
    CHECK(all_zero);

    // SineWave: frame 0 == 0 (sin(0)), signal present, bounded, mono
    // duplicated across both stereo channels. No tick ⇒ default gain 1.
    auto e2 = au::Engine::create();
    e2->register_cue(sine_cue("sine", 10.0f, 1000.0f, 1.0f));
    e2->play_2d("sine");
    for (float& x : buf) x = -3.0f;
    e2->render(buf, 64, 2);
    CHECK(buf[0] == 0.0f && buf[1] == 0.0f);         // t=0 ⇒ sin(0)=0
    bool present = false, bounded = true, stereo_dup = true;
    for (u32 f = 0; f < 64u; ++f) {
        const float l = buf[f*2 + 0];
        const float r = buf[f*2 + 1];
        if (l != 0.0f) present = true;
        if (l < -1.0f || l > 1.0f) bounded = false;
        if (l != r) stereo_dup = false;
    }
    CHECK(present);
    CHECK(bounded);
    CHECK(stereo_dup);
}

// ---- render: additive mix + hard soft-clip ------------------------
// Procedural cues with a CONSTANT generator make the mixed bus exact
// (no sin phase). No tick ⇒ default gain 1.0 per instance.
au::Cue const_cue(const char* id, float value) {
    au::Cue c;
    c.id = id;
    c.kind = au::CueKind::Procedural;
    c.duration_s = 100.0f;
    c.procedural = [value](float) { return value; };
    return c;
}

void test_render_mix() {
    auto e = au::Engine::create();
    e->register_cue(const_cue("c06", 0.6f));
    float buf[64];

    e->play_2d("c06");                               // 1 × 0.6 = 0.6
    e->render(buf, 16, 1);
    bool ok06 = true;
    for (u32 i = 0; i < 16u; ++i) if (!ap(buf[i], 0.6f)) ok06 = false;
    CHECK(ok06);

    e->play_2d("c06"); e->play_2d("c06");            // 3 × 0.6 = 1.8 → clip 1
    e->render(buf, 16, 1);
    bool clipHi = true;
    for (u32 i = 0; i < 16u; ++i) if (!ap(buf[i], 1.0f)) clipHi = false;
    CHECK(clipHi);

    auto e2 = au::Engine::create();
    e2->register_cue(const_cue("cneg", -0.7f));
    e2->play_2d("cneg"); e2->play_2d("cneg");        // -1.4 → clip -1
    e2->render(buf, 16, 2);
    bool clipLo = true;
    for (u32 i = 0; i < 32u; ++i) if (!ap(buf[i], -1.0f)) clipLo = false;
    CHECK(clipLo);
}

// ---- Procedural: t argument, pitch scaling, empty callback --------
void test_procedural() {
    auto e = au::Engine::create();
    au::Cue id; id.id = "id"; id.kind = au::CueKind::Procedural;
    id.duration_s = 100.0f;
    id.procedural = [](float t) { return t; };       // sample value == time
    e->register_cue(id);

    e->play_2d("id", au::kChannelSfx, 1.0f, 1.0f, false);   // pitch 1
    float buf[8];
    e->render(buf, 4, 1);
    const float inv = 1.0f / 48000.0f;               // default sample_rate
    CHECK(ap(buf[0], 0.0f) && ap(buf[1], inv) &&
          ap(buf[2], 2.0f * inv) && ap(buf[3], 3.0f * inv));

    auto e2 = au::Engine::create();
    e2->register_cue(id);
    e2->play_2d("id", au::kChannelSfx, 1.0f, 2.0f, false);  // pitch 2 ⇒ 2·t
    e2->render(buf, 4, 1);
    CHECK(ap(buf[1], 2.0f * inv) && ap(buf[3], 6.0f * inv));

    // Procedural with NO callback ⇒ silent.
    auto e3 = au::Engine::create();
    au::Cue np; np.id = "np"; np.kind = au::CueKind::Procedural;
    np.duration_s = 100.0f;                           // procedural left empty
    e3->register_cue(np);
    e3->play_2d("np");
    for (float& x : buf) x = 9.0f;
    e3->render(buf, 8, 1);
    bool zero = true;
    for (u32 i = 0; i < 8u; ++i) if (buf[i] != 0.0f) zero = false;
    CHECK(zero);
}

// ---- fade_in / fade_out gain ramps over ticks ---------------------
void test_fades() {
    auto e = au::Engine::create();
    e->register_cue(sine_cue("sine", 10.0f));
    const au::InstanceId id = e->play_2d("sine", au::kChannelSfx);
    e->fade_in(id, 1.0f);                              // resets play_head to 0
    e->tick(0.25f);                                   // 0.25/1.0
    CHECK(ap(e->active_instances()[0].final_attenuated_volume, 0.25f, 1e-4f));
    e->tick(0.25f);                                   // 0.50/1.0
    CHECK(ap(e->active_instances()[0].final_attenuated_volume, 0.50f, 1e-4f));
    e->tick(0.60f);                                   // 1.10 ≥ 1.0 ⇒ full
    CHECK(ap(e->active_instances()[0].final_attenuated_volume, 1.0f, 1e-4f));

    // fade_out: factor = t / max(1e-3, t+dt) where t = max(0, fade_out_s−dt)
    // and fade_out_s is updated to t each tick.
    auto e2 = au::Engine::create();
    e2->register_cue(sine_cue("sine", 10.0f));
    const au::InstanceId id2 = e2->play_2d("sine", au::kChannelSfx);
    e2->fade_out(id2, 1.0f);
    e2->tick(0.25f);                                  // t=.75, /1.0  = .75
    CHECK(ap(e2->active_instances()[0].final_attenuated_volume, 0.75f, 1e-4f));
    e2->tick(0.50f);                                  // t=.25, /.75  = .3333
    CHECK(ap(e2->active_instances()[0].final_attenuated_volume,
             0.25f / 0.75f, 1e-4f));
    e2->tick(0.50f);                                  // t=0 ⇒ faded → removed
    CHECK(e2->active_instances().empty());
}

// ---- non-finite control inputs must not poison instances / bus ----
// Regression: cardinal::clamp passes NaN THROUGH (NaN<lo and NaN>hi are
// both false), so play_3d's volume/pitch "sanitizing" clamps did NOT
// sanitize NaN. A NaN pitch made play_head_s NaN in tick(); the
// `play_head_s >= duration_s` expiry test is false for NaN so the
// instance was IMMORTAL, and render() then evaluated sin(2*pi*f*NaN) →
// a NaN output bus forever. A non-finite dt poisons the same way.
// Inputs must be forced finite (NaN vol→0/silent, NaN pitch→1,
// non-finite dt→no-op); every finite input stays bit-identical.
void test_nonfinite_inputs() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float inf  = std::numeric_limits<float>::infinity();

    {   // NaN pitch must not survive into the instance / play head.
        auto e = au::Engine::create();
        e->register_cue(sine_cue("sine", 1.0f));
        const au::InstanceId id = e->play_2d("sine", au::kChannelSfx,
                                              1.0f, qnan, false);
        CHECK(id != 0u);
        auto a = e->active_instances();
        CHECK(a.size() == 1u);
        if (!a.empty()) {
            CHECK(a[0].pitch == a[0].pitch);             // finite, not NaN
            CHECK(a[0].pitch >= 0.05f && a[0].pitch <= 8.0f);
        }
        e->tick(0.25f);                                  // finite dt
        a = e->active_instances();
        if (!a.empty()) {
            CHECK(a[0].play_head_s == a[0].play_head_s); // finite
            CHECK(ap(a[0].play_head_s, 0.25f));          // pitch→1
        }
        e->tick(1.0f);                                   // 1.25 ≥ 1.0
        CHECK(e->active_instances().empty());            // expires, not immortal
    }
    {   // NaN volume → silent (0), not a NaN bus.
        auto e = au::Engine::create();
        e->register_cue(sine_cue("sine", 1.0f));
        e->play_2d("sine", au::kChannelSfx, qnan, 1.0f, false);
        e->tick(0.01f);
        auto a = e->active_instances();
        CHECK(a.size() == 1u);
        if (!a.empty()) {
            CHECK(a[0].volume == a[0].volume);                       // finite
            CHECK(a[0].final_attenuated_volume ==
                  a[0].final_attenuated_volume);                     // finite
            CHECK(ap(a[0].final_attenuated_volume, 0.0f, 1e-4f));    // silent
        }
        float buf[32];
        for (float& x : buf) x = 7.0f;
        e->render(buf, 16, 1);
        bool finite_bus = true;
        for (u32 i = 0; i < 16u; ++i)
            if (!(buf[i] == buf[i])) finite_bus = false;             // no NaN
        CHECK(finite_bus);
    }
    {   // set_channel_volume(NaN/+Inf/-Inf): cardinal::clamp passes
        // NaN through, so the existing clamp at audio.cpp was NaN-blind.
        // Without the isfinite guard, channel volume → NaN → render
        // bus → NaN. Verify the new guard sets channel to silent (0).
        auto e = au::Engine::create();
        e->register_cue(sine_cue("sine", 1.0f));
        e->set_channel_volume(au::kChannelSfx, qnan);
        const au::Channel* c = e->channel(au::kChannelSfx);
        CHECK(c != nullptr);
        if (c) CHECK(ap(c->volume, 0.0f));
        e->set_channel_volume(au::kChannelSfx,  inf);
        if ((c = e->channel(au::kChannelSfx))) CHECK(ap(c->volume, 0.0f));
        e->set_channel_volume(au::kChannelSfx, -inf);
        if ((c = e->channel(au::kChannelSfx))) CHECK(ap(c->volume, 0.0f));
        // A subsequent finite call restores to a defined value.
        e->set_channel_volume(au::kChannelSfx, 0.7f);
        if ((c = e->channel(au::kChannelSfx))) CHECK(ap(c->volume, 0.7f));
    }
    {   // set_emitter_position: a NaN component MUST be rejected
        // (caller bug must not poison the stored instance state). The
        // previous position is preserved.
        auto e = au::Engine::create();
        e->register_cue(sine_cue("sine", 1.0f));
        const au::InstanceId id = e->play_3d("sine", Vec3{5, 0, 0},
                                              au::kChannelSfx, 1.0f, 1.0f, false);
        CHECK(id != 0u);
        e->set_emitter_position(id, Vec3{qnan, 0, 0});       // rejected
        auto a = e->active_instances();
        CHECK(a.size() == 1u);
        if (!a.empty()) {
            CHECK(ap(a[0].position.x, 5.0f));                 // unchanged
            CHECK(ap(a[0].position.y, 0.0f));
            CHECK(ap(a[0].position.z, 0.0f));
        }
        e->set_emitter_position(id, Vec3{0, inf, 0});        // rejected
        e->set_emitter_position(id, Vec3{0, 0, qnan});       // rejected
        a = e->active_instances();
        if (!a.empty()) CHECK(ap(a[0].position.x, 5.0f));    // still unchanged
        // A finite triple goes through.
        e->set_emitter_position(id, Vec3{1, 2, 3});
        a = e->active_instances();
        if (!a.empty()) {
            CHECK(ap(a[0].position.x, 1.0f));
            CHECK(ap(a[0].position.y, 2.0f));
            CHECK(ap(a[0].position.z, 3.0f));
        }
    }
    {   // set_listener: a NaN component in position/forward/up MUST
        // be rejected (caller bug must not poison every 3D-attenuated
        // instance simultaneously). The previous good listener is
        // preserved.
        auto e = au::Engine::create();
        e->register_cue(sine_cue("sine", 1.0f));
        au::Listener L_good; L_good.position = {10, 0, 0};
        e->set_listener(L_good);

        // Now feed a bad listener — position NaN.
        au::Listener L_bad = L_good;
        L_bad.position.x = qnan;
        e->set_listener(L_bad);                          // rejected
        // forward NaN
        L_bad = L_good; L_bad.forward.y = qnan;
        e->set_listener(L_bad);                          // rejected
        // up NaN
        L_bad = L_good; L_bad.up.z = qnan;
        e->set_listener(L_bad);                          // rejected
        // +Inf component
        L_bad = L_good; L_bad.position.z = inf;
        e->set_listener(L_bad);                          // rejected

        // Listener should still be the good one. Verify by playing at
        // (10,0,0) — distance 0, attenuation 1.0. If listener had been
        // poisoned to NaN, the attenuation would be 0 (defensive guard
        // in compute_3d_attenuation_) or NaN (without the guard).
        e->play_3d("sine", {10, 0, 0}, au::kChannelSfx);
        e->tick(0.01f);
        auto a = e->active_instances();
        CHECK(a.size() == 1u);
        if (!a.empty())
            CHECK(ap(a[0].final_attenuated_volume, 1.0f, 1e-4f));

        // A subsequent fully-finite Listener update goes through.
        au::Listener L2; L2.position = {100, 0, 0};
        e->set_listener(L2);
        // The already-playing instance at (10,0,0) is now far from the
        // new listener — distance 90 ≥ max (50) ⇒ attenuation 0.
        e->tick(0.01f);
        a = e->active_instances();
        if (!a.empty())
            CHECK(ap(a[0].final_attenuated_volume, 0.0f, 1e-4f));
    }
    {   // register_cue must sanitize Cue float fields — duration_s,
        // sine_frequency_hz, gain. NaN duration_s would make the
        // instance immortal (`play_head_s >= duration_s` unordered-
        // false). NaN sine_frequency_hz or gain would render NaN
        // samples to the audio bus. Each field is sanitized to the
        // struct's compile-time default.
        auto e = au::Engine::create();
        au::Cue bad;
        bad.id                = "bad";
        bad.kind              = au::CueKind::SineWave;
        bad.duration_s        = qnan;
        bad.sine_frequency_hz = qnan;
        bad.gain              = inf;
        e->register_cue(bad);
        const au::Cue* c = e->find_cue("bad");
        CHECK(c != nullptr);
        if (c) {
            // Sanitized to compile-time defaults at registration.
            CHECK(ap(c->duration_s,        1.0f));
            CHECK(ap(c->sine_frequency_hz, 440.0f));
            CHECK(ap(c->gain,              1.0f));
        }
        // Sanity: play through the cue — instance must expire on its
        // sanitized 1.0s duration AND render samples must be finite
        // (no NaN bus).
        e->play_2d("bad", au::kChannelSfx, 1.0f, 1.0f, false);
        e->tick(0.5f);
        CHECK(e->active_instances().size() == 1u);   // alive at 0.5
        float buf[32]; for (float& x : buf) x = 7.0f;
        e->render(buf, 16, 1);
        bool finite_bus = true;
        for (u32 i = 0; i < 16u; ++i)
            if (!(buf[i] == buf[i])) finite_bus = false;
        CHECK(finite_bus);
        e->tick(1.0f);
        // 0.5 + 1.0 ≥ 1.0 → expires (would be immortal without the fix).
        CHECK(e->active_instances().empty());
    }
    {   // Non-finite dt is a no-op; the next valid tick is correct.
        auto e = au::Engine::create();
        e->register_cue(sine_cue("sine", 1.0f));
        e->play_2d("sine", au::kChannelSfx, 1.0f, 1.0f, false);
        e->tick(qnan);
        auto a = e->active_instances();
        CHECK(a.size() == 1u);
        if (!a.empty()) CHECK(ap(a[0].play_head_s, 0.0f));   // untouched
        e->tick(inf);
        a = e->active_instances();
        if (!a.empty()) CHECK(ap(a[0].play_head_s, 0.0f));
        e->tick(0.5f);                                        // valid
        a = e->active_instances();
        CHECK(a.size() == 1u);
        if (!a.empty()) CHECK(ap(a[0].play_head_s, 0.5f));    // not NaN
    }
}

}  // namespace

int main() {
    test_channels();
    test_cues();
    test_play_control();
    test_instance_cull();
    test_tick();
    test_attenuation();
    test_render();
    test_render_mix();
    test_procedural();
    test_fades();
    test_nonfinite_inputs();

    if (g_fail == 0) {
        cardinal::log::infof("audiotest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("audiotest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
