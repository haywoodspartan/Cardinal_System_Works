// =============================================================================
// Cardinal — deterministic keyframe-animation regression suite.
//
// Curve<T>::sample + Track/Clip/Player are the animation backbone the
// sequencer, curve editor and cinematics drive. A regression in key
// sorting, segment interpolation (the LEFT key's mode governs its
// segment), the lower_bound boundary, or wrap (Clamp/Loop/PingPong)
// silently corrupts every animated property. Pure CPU + fully
// deterministic; exact small-curve values and the cubic-Hermite basis
// are pinned by hand-computed constants (no <cmath> in the test). Exit
// 0 = all pass.
// =============================================================================

#include <cardinal/anim/anim.hpp>
#include <cardinal/core/log.hpp>

#include <limits>
#include <memory>

namespace {

namespace an = cardinal::anim;
using an::InterpMode;
using an::WrapMode;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("animtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// ---- add_key sorted insert / empty / single / duration -------------
void test_curve_basics() {
    an::Curve<float> c;
    CHECK(c.empty());
    CHECK(ap(c.duration(), 0.0f));
    CHECK(ap(c.sample(5.0f), 0.0f));                  // empty → T{}

    c.add_key(2.0f, 2.0f);                            // out-of-order adds
    c.add_key(0.0f, 0.0f);
    c.add_key(1.0f, 1.0f);
    CHECK(c.keys.size() == sz(3));
    CHECK(ap(c.keys[0].time, 0.0f) && ap(c.keys[1].time, 1.0f) &&
          ap(c.keys[2].time, 2.0f));                  // sorted by time
    CHECK(ap(c.keys[0].value, 0.0f) && ap(c.keys[2].value, 2.0f));
    CHECK(ap(c.duration(), 2.0f));                    // == back().time
    c.add_key(1.0f, 99.0f);                           // duplicate time kept
    CHECK(c.keys.size() == sz(4));
    bool monotone = true;
    for (cardinal::usize i = 1; i < c.keys.size(); ++i)
        if (c.keys[i].time < c.keys[i-1].time) monotone = false;
    CHECK(monotone);
    c.clear();
    CHECK(c.empty() && ap(c.duration(), 0.0f));

    an::Curve<float> one;
    one.add_key(3.0f, 7.5f);
    CHECK(ap(one.sample(-9.0f), 7.5f));               // single key: const
    CHECK(ap(one.sample(3.0f), 7.5f));
    CHECK(ap(one.sample(100.0f), 7.5f));
}

// ---- Linear / Step / clamp -----------------------------------------
void test_linear_step() {
    an::Curve<float> lin;
    lin.add_key(0.0f, 10.0f, InterpMode::Linear);
    lin.add_key(10.0f, 20.0f, InterpMode::Linear);
    CHECK(ap(lin.sample(0.0f), 10.0f));
    CHECK(ap(lin.sample(10.0f), 20.0f));
    CHECK(ap(lin.sample(5.0f), 15.0f));
    CHECK(ap(lin.sample(2.5f), 12.5f));
    CHECK(ap(lin.sample(-5.0f), 10.0f));              // clamp to front
    CHECK(ap(lin.sample(100.0f), 20.0f));             // clamp to back

    an::Curve<float> stp;
    stp.add_key(0.0f, 1.0f, InterpMode::Step);
    stp.add_key(10.0f, 2.0f, InterpMode::Step);
    CHECK(ap(stp.sample(0.0f), 1.0f));
    CHECK(ap(stp.sample(5.0f), 1.0f));                // hold k0.value
    CHECK(ap(stp.sample(9.999f), 1.0f));
    CHECK(ap(stp.sample(10.0f), 2.0f));               // >= back → back
    CHECK(ap(stp.sample(-3.0f), 1.0f));               // clamp → front
}

// ---- the LEFT key's mode governs its segment + lower_bound edge -----
void test_segment_mode() {
    an::Curve<float> c;
    c.add_key(0.0f,  0.0f, InterpMode::Step);    // segment [0,10): Step
    c.add_key(10.0f, 10.0f, InterpMode::Linear); // segment [10,20): Linear
    c.add_key(20.0f, 30.0f, InterpMode::Linear);
    CHECK(ap(c.sample(5.0f), 0.0f));             // Step holds k0.value
    // At the interior boundary t==10 the lower_bound lands on the LEFT
    // segment at u=1 → Step still returns k0.value (0), NOT key1's 10.
    CHECK(ap(c.sample(10.0f), 0.0f));
    CHECK(ap(c.sample(15.0f), 20.0f));           // Linear lerp(10,30,.5)
    CHECK(ap(c.sample(20.0f), 30.0f));           // >= back → back
}

// ---- CubicHermite (basis pinned by hand-computed constants) --------
void test_hermite() {
    // A: v0=0 v1=1 m0=0 m1=0  ⇒  value = -2u³ + 3u²
    {
        an::Curve<float> c;
        c.add_key(0.0f, 0.0f, InterpMode::CubicHermite);
        c.add_key(1.0f, 1.0f, InterpMode::CubicHermite);
        c.keys[0].out_tangent = 0.0f;
        c.keys[1].in_tangent  = 0.0f;
        CHECK(ap(c.sample(0.5f),  0.5f));        // -2(.125)+3(.25)
        CHECK(ap(c.sample(0.25f), 0.15625f));    // -2(.0156)+3(.0625)
        CHECK(ap(c.sample(0.0f),  0.0f));        // front guard
        CHECK(ap(c.sample(1.0f),  1.0f));        // back guard
    }
    // B: v0=0 v1=0 m0=1 m1=1  ⇒  value = 2u³ - 3u² + u
    {
        an::Curve<float> c;
        c.add_key(0.0f, 0.0f, InterpMode::CubicHermite);
        c.add_key(1.0f, 0.0f, InterpMode::CubicHermite);
        c.keys[0].out_tangent = 1.0f;
        c.keys[1].in_tangent  = 1.0f;
        CHECK(ap(c.sample(0.5f),  0.0f));        // .25-.75+.5
        CHECK(ap(c.sample(0.25f), 0.09375f));    // .03125-.1875+.25
    }
}

// ---- WrapMode: Clamp / Loop / PingPong (exact) ---------------------
void test_wrap() {
    auto mk = [] {
        an::Curve<float> c;
        c.add_key(0.0f,  0.0f, InterpMode::Linear);
        c.add_key(10.0f, 10.0f, InterpMode::Linear);   // value == time
        return c;
    };
    an::Curve<float> clamp = mk();                     // default Clamp
    CHECK(ap(clamp.sample(-100.0f), 0.0f));
    CHECK(ap(clamp.sample(1.0e6f), 10.0f));

    an::Curve<float> loop = mk();
    loop.wrap = WrapMode::Loop;
    CHECK(ap(loop.sample(12.0f), 2.0f));               // fmod(12,10)=2
    CHECK(ap(loop.sample(-2.0f), 8.0f));               // -2 → +8
    CHECK(ap(loop.sample(25.0f), 5.0f));
    CHECK(ap(loop.sample(10.0f), 0.0f));               // wraps to 0, not 10
    CHECK(ap(loop.sample(20.0f), 0.0f));

    an::Curve<float> pp = mk();
    pp.wrap = WrapMode::PingPong;
    CHECK(ap(pp.sample(15.0f), 5.0f));                 // 20-15
    CHECK(ap(pp.sample(10.0f), 10.0f));                // peak
    CHECK(ap(pp.sample(25.0f), 5.0f));
    CHECK(ap(pp.sample(-5.0f), 5.0f));
    CHECK(ap(pp.sample(30.0f), 10.0f));
}

// ---- Vec3 lerp specialisation --------------------------------------
void test_vec3_curve() {
    an::Curve<cardinal::scene::Vec3> c;
    const cardinal::scene::Vec3 a{ 0.0f, 0.0f, 0.0f };
    const cardinal::scene::Vec3 b{ 10.0f, 20.0f, 30.0f };
    c.add_key(0.0f,  a, InterpMode::Linear);
    c.add_key(10.0f, b, InterpMode::Linear);
    const cardinal::scene::Vec3 m = c.sample(5.0f);
    CHECK(ap(m.x, 5.0f) && ap(m.y, 10.0f) && ap(m.z, 15.0f));
    const cardinal::scene::Vec3 e0 = c.sample(-1.0f);
    CHECK(ap(e0.x, 0.0f) && ap(e0.z, 0.0f));           // clamp front
}

// ---- Track / Clip / Player -----------------------------------------
void test_track_clip_player() {
    // Track: writer receives the sampled value; empty curve → no write.
    {
        float dst = 42.0f;
        an::Track<float> tr("t", [&](const float& v) { dst = v; });
        tr.apply(0.0f);                                // empty → no-op
        CHECK(ap(dst, 42.0f));
        tr.curve().add_key(0.0f, 10.0f, InterpMode::Linear);
        tr.curve().add_key(10.0f, 20.0f, InterpMode::Linear);
        tr.apply(5.0f);
        CHECK(ap(dst, 15.0f));
        CHECK(ap(tr.duration(), 10.0f));
        CHECK(tr.name() == "t");
    }

    // Clip duration == max(track durations); apply drives all tracks.
    {
        float da = -1.0f, db = -1.0f;
        auto ta = std::make_unique<an::Track<float>>(
            "A", [&](const float& v) { da = v; });
        auto tb = std::make_unique<an::Track<float>>(
            "B", [&](const float& v) { db = v; });
        ta->curve().add_key(0.0f, 0.0f, InterpMode::Linear);
        ta->curve().add_key(5.0f, 50.0f, InterpMode::Linear);   // dur 5
        tb->curve().add_key(0.0f, 0.0f, InterpMode::Linear);
        tb->curve().add_key(8.0f, 80.0f, InterpMode::Linear);   // dur 8
        an::Clip clip("c");
        clip.add_track(std::move(ta));
        clip.add_track(std::move(tb));
        CHECK(ap(clip.duration(), 8.0f));              // max
        clip.apply(4.0f);
        CHECK(ap(da, 40.0f) && ap(db, 40.0f));
    }

    // Player: tick advances time*speed; loop off clamps + stops; loop
    // on wraps; not-playing / null-clip are no-ops.
    {
        float dst = -1.0f;
        auto tr = std::make_unique<an::Track<float>>(
            "p", [&](const float& v) { dst = v; });
        tr->curve().add_key(0.0f, 0.0f, InterpMode::Linear);
        tr->curve().add_key(10.0f, 100.0f, InterpMode::Linear);
        auto clip = std::make_shared<an::Clip>("clip");
        clip->add_track(std::move(tr));

        an::Player p(clip);
        CHECK(ap(p.duration(), 10.0f));
        p.tick(1.0f);                                  // not playing → no-op
        CHECK(ap(p.time(), 0.0f) && ap(dst, -1.0f));
        p.play();
        p.tick(2.0f);
        CHECK(ap(p.time(), 2.0f) && ap(dst, 20.0f));
        p.set_speed(2.0f);
        p.tick(1.0f);                                  // +1*2 → t=4
        CHECK(ap(p.time(), 4.0f) && ap(dst, 40.0f));
        // Loop off: overrun clamps to duration and stops.
        p.set_speed(1.0f);
        p.seek(9.0f);
        p.tick(5.0f);                                  // 14 > 10
        CHECK(ap(p.time(), 10.0f) && ap(dst, 100.0f));
        CHECK(!p.playing());

        // Loop on: overrun wraps via fmod, keeps playing.
        float d2 = -1.0f;
        auto tr2 = std::make_unique<an::Track<float>>(
            "p2", [&](const float& v) { d2 = v; });
        tr2->curve().add_key(0.0f, 0.0f, InterpMode::Linear);
        tr2->curve().add_key(10.0f, 100.0f, InterpMode::Linear);
        auto clip2 = std::make_shared<an::Clip>("c2");
        clip2->add_track(std::move(tr2));
        an::Player pl(clip2);
        pl.set_loop(true);
        pl.play();
        pl.seek(9.0f);
        pl.tick(5.0f);                                 // 14 → fmod → 4
        CHECK(ap(pl.time(), 4.0f) && ap(d2, 40.0f));
        CHECK(pl.playing());

        // Null clip → tick is a safe no-op.
        an::Player pn;
        pn.play();
        pn.tick(1.0f);
        CHECK(ap(pn.duration(), 0.0f));
        CHECK(ap(pn.time(), 0.0f));
    }
}

// ---- non-finite sample time must not OOB (NaN/Inf robustness) ------
// Regression: for any finite t, if lower_bound() returns begin() then the
// `t <= front` guard already returned, so i1 >= 1. Only a non-finite t
// (NaN compares unordered → both clamp guards false) drives lower_bound to
// begin(), i1=0, i0=i1-1 → SIZE_MAX → keys[SIZE_MAX] OOB read in the
// noexcept sample(). Defined behaviour: clamp-to-front, never an OOB.
void test_nonfinite_time() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float inf  = std::numeric_limits<float>::infinity();

    an::Curve<float> c;
    c.add_key(0.0f,  0.0f, InterpMode::Linear);
    c.add_key(10.0f, 10.0f, InterpMode::Linear);   // value == time

    CHECK(ap(c.sample(qnan),  0.0f));              // NaN → clamp-to-front
    CHECK(ap(c.sample(-inf),  0.0f));              // Clamp: −inf < 0 → front
    CHECK(ap(c.sample( inf), 10.0f));              // Clamp: inf > dur → back

    // Loop / PingPong fmod() of a non-finite time yields NaN internally —
    // still a defined in-gamut value, never an OOB read.
    an::Curve<float> lp = c; lp.wrap = WrapMode::Loop;
    CHECK(ap(lp.sample(qnan), 0.0f));
    CHECK(ap(lp.sample(inf),  0.0f));              // fmod(inf,dur)=NaN→front
    an::Curve<float> pp = c; pp.wrap = WrapMode::PingPong;
    CHECK(ap(pp.sample(qnan), 0.0f));
    CHECK(ap(pp.sample(inf),  0.0f));

    // Player fed a NaN dt: the Player::tick GUARD (root-cause fix —
    // companion to d8153cc's downstream Curve::sample defense) skips
    // the bad frame entirely. time_ is NOT poisoned, clip_->apply is
    // NOT called, the track's setter is NOT invoked — `dst` keeps its
    // pre-tick value. Subsequent finite ticks resume normally.
    float dst = -1.0f;
    auto tr = std::make_unique<an::Track<float>>(
        "n", [&](const float& v) { dst = v; });
    tr->curve().add_key(0.0f,   0.0f, InterpMode::Linear);
    tr->curve().add_key(10.0f, 100.0f, InterpMode::Linear);
    auto clip = std::make_shared<an::Clip>("c");
    clip->add_track(std::move(tr));
    an::Player p(clip);
    p.play();
    p.tick(qnan);                                  // skipped — no apply call
    CHECK(ap(dst, -1.0f));                          // sentinel untouched
    p.tick(0.5f);                                   // resume: time_ = 0.5
    CHECK(ap(dst, 5.0f));                           // sample(0.5) on a 0..10 → 0..100 ramp
}

// ---- Player::seek and Player::set_speed must reject non-finite -----
// Both feed the `time_ += dt * speed_` accumulator in tick(). A NaN
// time_ (from seek) or speed_ (from set_speed) permanently poisons the
// accumulator. Curve::sample's downstream defenses (d8153cc) keep the
// SAMPLED value defined (clamp-to-front on NaN, clamp-to-back on Inf
// for Clamp wrap), but the accumulator itself stays poisoned →
// animation frozen forever. This is the SECOND-INGRESS pattern, same
// family as sim::set_time_scale (70a9324) and Sky::set_hour /
// set_day_length_seconds (this commit).
void test_player_setter_nonfinite() {
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float inf  = std::numeric_limits<float>::infinity();

    // Build a Player with a 0..10 → 0..100 ramp Curve.
    float dst = -1.0f;
    auto tr = std::make_unique<an::Track<float>>(
        "n", [&](const float& v) { dst = v; });
    tr->curve().add_key(0.0f,   0.0f, InterpMode::Linear);
    tr->curve().add_key(10.0f, 100.0f, InterpMode::Linear);
    auto clip = std::make_shared<an::Clip>("c");
    clip->add_track(std::move(tr));
    an::Player p(clip);
    p.play();

    // seek(NaN) → reset to 0 (not poisoned). A finite tick now samples
    // from time 0 → expect dst == 0.0.
    p.seek(qnan);
    CHECK(ap(p.time(), 0.0f));
    p.tick(0.0f);                        // no time advance, just apply
    CHECK(ap(dst, 0.0f));

    // seek(+Inf), seek(-Inf) — also route to 0.
    p.seek( inf); CHECK(ap(p.time(), 0.0f));
    p.seek(-inf); CHECK(ap(p.time(), 0.0f));

    // set_speed(NaN) → reset to 0. tick(0.5) with speed=0 advances
    // time_ by 0 (no change), so dst stays at front (0).
    p.set_speed(qnan);
    CHECK(ap(p.speed(), 0.0f));
    p.tick(0.5f);
    CHECK(ap(p.time(),  0.0f));
    CHECK(ap(dst,       0.0f));

    // set_speed(+Inf) / set_speed(-Inf) → also 0.
    p.set_speed( inf); CHECK(ap(p.speed(), 0.0f));
    p.set_speed(-inf); CHECK(ap(p.speed(), 0.0f));

    // After resetting to a normal speed, animation advances normally
    // (proves the prior NaN didn't leak into a future state).
    p.set_speed(1.0f);
    p.tick(0.5f);
    CHECK(ap(p.time(), 0.5f));
    CHECK(ap(dst,      5.0f));           // 0.5 on a 0..10 → 0..100 ramp
}

}  // namespace

int main() {
    test_curve_basics();
    test_linear_step();
    test_segment_mode();
    test_hermite();
    test_wrap();
    test_vec3_curve();
    test_track_clip_player();
    test_nonfinite_time();
    test_player_setter_nonfinite();

    if (g_fail == 0) {
        cardinal::log::infof("animtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("animtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
