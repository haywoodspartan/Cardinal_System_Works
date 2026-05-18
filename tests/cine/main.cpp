// =============================================================================
// Cardinal — deterministic cinematic-sequencer regression suite.
//
// cine::Player::tick advances the play head and EDGE-TRIGGERS track
// events exactly once as the head crosses each event time. The pinned
// contract:
//
//   * window is (prev, now]  — exclusive lower, inclusive upper, so an
//     event AT the start time never fires and one AT the landing time
//     does; each event fires exactly once.
//   * loop wrap: a tick that runs past duration applies a TAIL pass over
//     (prev, duration] using the local tick-start time, then resets and
//     continues a MAIN pass over (0, carry] using the member tracker.
//   * non-looping: clamp to duration, fire events up to & including it,
//     stop playing, then freeze (no-op ticks).
//   * seek clamps to [0, duration]; play/pause/stop; speed scales dt;
//     speed 0 freezes; tick() (unlike seek) does not lower-clamp.
//   * null Sequence and unset callbacks are safe no-ops.
//
// Event + Camera tracks are observed via recording callbacks; Audio
// (needs an audio::Engine) and Animation (needs an anim::Player) tracks
// share the identical window predicate and are out of scope. All times
// are small exact floats — no FP-boundary fragility. Exit 0 = all pass.
// =============================================================================

#include <cardinal/cine/cine.hpp>
#include <cardinal/core/log.hpp>

#include <string>
#include <utility>          // std::move
#include <vector>

namespace {

namespace cn = cardinal::cine;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("cinetest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }

// Recording sink for the two host callbacks.
struct Rec {
    std::vector<std::string>   ev;     // event names, in fire order
    std::vector<cardinal::u32> cam;    // camera actor ids, in fire order
    std::vector<float>         blend;  // matching blend seconds
};
int count_occ(const std::vector<std::string>& v, const char* s) {
    int n = 0;
    for (const auto& x : v) if (x == s) ++n;
    return n;
}
bool fired(const std::vector<std::string>& v, const char* s) {
    return count_occ(v, s) > 0;
}

cn::EventTrack ev_track(std::vector<cn::EventCall> evs) {
    cn::EventTrack t;
    t.name = "ev";
    t.events = std::move(evs);
    return t;
}
void wire(cn::Player& p, Rec& r) {
    p.set_event_dispatch([&r](const std::string& n, const std::string&) {
        r.ev.push_back(n);
    });
    p.set_camera_switch([&r](cardinal::u32 a, float b) {
        r.cam.push_back(a);
        r.blend.push_back(b);
    });
}

// ---- play / pause / stop / null-Sequence safety --------------------
void test_play_state() {
    cn::Sequence seq; seq.duration_s = 5.0f;
    cn::Player p(&seq);
    CHECK(p.sequence() == &seq);
    CHECK(!seq.playing);                 // default

    p.play();  CHECK(seq.playing);
    p.pause(); CHECK(!seq.playing);
    p.play();  CHECK(seq.playing);

    p.seek(3.0f);
    CHECK(ap(seq.play_head_s, 3.0f));
    p.stop();
    CHECK(!seq.playing);                 // stop → paused + rewound
    CHECK(ap(seq.play_head_s, 0.0f));

    // Null-Sequence Player: every entry point is a safe no-op.
    cn::Player pn(nullptr);
    CHECK(pn.sequence() == nullptr);
    pn.play(); pn.pause(); pn.stop(); pn.seek(2.0f); pn.tick(1.0f);
    CHECK(true);                         // reached without crashing
}

// ---- seek clamps to [0, duration] ----------------------------------
void test_seek_clamp() {
    cn::Sequence seq; seq.duration_s = 5.0f;
    cn::Player p(&seq);
    p.seek(2.5f);   CHECK(ap(seq.play_head_s, 2.5f));
    p.seek(-3.0f);  CHECK(ap(seq.play_head_s, 0.0f));   // clamp low
    p.seek(99.0f);  CHECK(ap(seq.play_head_s, 5.0f));   // clamp high
    p.seek(5.0f);   CHECK(ap(seq.play_head_s, 5.0f));   // exact upper
    p.seek(0.0f);   CHECK(ap(seq.play_head_s, 0.0f));
}

// ---- tick while paused is a no-op ----------------------------------
void test_tick_paused_noop() {
    cn::Sequence seq; seq.duration_s = 10.0f;
    seq.tracks.push_back(ev_track({ {2.0f, "e2", ""} }));
    cn::Player p(&seq); Rec r; wire(p, r);
    p.seek(0.0f);
    p.tick(5.0f);                         // not playing → nothing happens
    CHECK(ap(seq.play_head_s, 0.0f));
    CHECK(r.ev.empty());
}

// ---- edge trigger: exclusive lower, inclusive upper, fire-once -----
void test_event_edge_trigger() {
    cn::Sequence seq;
    seq.duration_s = 10.0f;
    seq.looping = false;
    seq.speed   = 1.0f;
    seq.tracks.push_back(ev_track({
        {0.0f, "e0", ""}, {2.0f, "e2", ""},
        {5.0f, "e5", ""}, {8.0f, "e8", ""},
    }));
    cn::Player p(&seq); Rec r; wire(p, r);
    p.seek(0.0f); p.play();

    p.tick(2.0f);                         // (0,2] → e2 (==t, inclusive)
    CHECK(r.ev.size() == sz(1));
    CHECK(r.ev[0] == "e2");
    CHECK(ap(seq.play_head_s, 2.0f));
    CHECK(!fired(r.ev, "e0"));            // event @0 == start → never (excl. lower)

    p.tick(3.0f);                         // (2,5] → e5
    CHECK(r.ev.size() == sz(2));
    CHECK(r.ev[1] == "e5");

    p.tick(2.0f);                         // (5,7] → nothing
    CHECK(r.ev.size() == sz(2));

    p.tick(2.0f);                         // (7,9] → e8
    CHECK(r.ev.size() == sz(3));
    CHECK(r.ev[2] == "e8");

    p.tick(0.5f);                         // (9,9.5] → nothing; no refire
    CHECK(r.ev.size() == sz(3));
    CHECK(count_occ(r.ev, "e2") == 1);    // each event fired exactly once
    CHECK(count_occ(r.ev, "e8") == 1);
    CHECK(!fired(r.ev, "e0"));
}

// ---- loop wrap: tail pass + carry continuation ---------------------
void test_loop_wrap() {
    cn::Sequence seq;
    seq.duration_s = 5.0f;
    seq.looping = true;
    seq.speed   = 1.0f;
    seq.tracks.push_back(ev_track({
        {0.0f, "z", ""},   // at 0 → never (excl. lower), even across wrap
        {1.0f, "a", ""},
        {4.0f, "b", ""},
    }));
    cn::Player p(&seq); Rec r; wire(p, r);
    p.seek(0.0f); p.play();

    p.tick(3.0f);                         // (0,3] → "a"
    CHECK(r.ev.size() == sz(1));
    CHECK(r.ev[0] == "a");
    CHECK(ap(seq.play_head_s, 3.0f));

    // prev=3, t=6 >= 5 → wrap. carry = 1.
    //   tail pass (prev=3, dur=5]  → "b"
    //   reset, main pass (0, carry=1] → "a" (new lap)
    p.tick(3.0f);
    CHECK(r.ev.size() == sz(3));
    CHECK(r.ev[1] == "b");                // tail pass uses tick-start prev
    CHECK(r.ev[2] == "a");                // main pass after reset
    CHECK(ap(seq.play_head_s, 1.0f));     // continued from carry
    CHECK(!fired(r.ev, "z"));             // @0 never, even on the wrap lap

    // Event exactly AT duration fires in the tail pass (inclusive upper).
    cn::Sequence s2;
    s2.duration_s = 5.0f; s2.looping = true; s2.speed = 1.0f;
    s2.tracks.push_back(ev_track({ {5.0f, "end", ""} }));
    cn::Player p2(&s2); Rec r2; wire(p2, r2);
    p2.seek(0.0f); p2.play();
    p2.tick(3.0f);                        // (0,3] → none
    CHECK(r2.ev.empty());
    p2.tick(3.0f);                        // wrap: tail (3,5] → "end"
    CHECK(r2.ev.size() == sz(1));
    CHECK(r2.ev[0] == "end");
    CHECK(ap(s2.play_head_s, 1.0f));
}

// ---- non-looping: clamp to duration, stop, then freeze -------------
void test_non_looping_clamp() {
    cn::Sequence seq;
    seq.duration_s = 5.0f;
    seq.looping = false;
    seq.speed   = 1.0f;
    seq.tracks.push_back(ev_track({ {2.0f, "m", ""}, {5.0f, "end", ""} }));
    cn::Player p(&seq); Rec r; wire(p, r);
    p.seek(0.0f); p.play();

    p.tick(3.0f);                         // (0,3] → "m"
    CHECK(r.ev.size() == sz(1));
    CHECK(r.ev[0] == "m");

    p.tick(10.0f);                        // overshoot → clamp 5, stop;
                                          // (3,5] → "end" (incl. upper)
    CHECK(r.ev.size() == sz(2));
    CHECK(r.ev[1] == "end");
    CHECK(ap(seq.play_head_s, 5.0f));
    CHECK(!seq.playing);                  // auto-stopped at the end

    p.tick(1.0f);                         // frozen: stopped → no-op
    CHECK(r.ev.size() == sz(2));
    CHECK(ap(seq.play_head_s, 5.0f));
}

// ---- speed scaling (incl. 0 freeze and backward) -------------------
void test_speed() {
    {   // speed 2 → head moves twice as fast.
        cn::Sequence seq;
        seq.duration_s = 100.0f; seq.looping = false; seq.speed = 2.0f;
        seq.tracks.push_back(ev_track({ {10.0f, "x", ""} }));
        cn::Player p(&seq); Rec r; wire(p, r);
        p.seek(0.0f); p.play();
        p.tick(3.0f);                     // t = 6
        CHECK(r.ev.empty());
        CHECK(ap(seq.play_head_s, 6.0f));
        p.tick(3.0f);                     // t = 12 → crosses 10
        CHECK(r.ev.size() == sz(1));
        CHECK(r.ev[0] == "x");
        CHECK(ap(seq.play_head_s, 12.0f));
    }
    {   // speed 0.5 → half rate.
        cn::Sequence seq;
        seq.duration_s = 100.0f; seq.looping = false; seq.speed = 0.5f;
        seq.tracks.push_back(ev_track({ {2.0f, "y", ""} }));
        cn::Player p(&seq); Rec r; wire(p, r);
        p.seek(0.0f); p.play();
        p.tick(3.0f);                     // t = 1.5
        CHECK(r.ev.empty());
        CHECK(ap(seq.play_head_s, 1.5f));
        p.tick(3.0f);                     // t = 3 → crosses 2
        CHECK(r.ev.size() == sz(1));
        CHECK(r.ev[0] == "y");
        CHECK(ap(seq.play_head_s, 3.0f));
    }
    {   // speed 0 → fully frozen, still "playing", no events.
        cn::Sequence seq;
        seq.duration_s = 10.0f; seq.looping = false; seq.speed = 0.0f;
        seq.tracks.push_back(ev_track({ {1.0f, "z", ""} }));
        cn::Player p(&seq); Rec r; wire(p, r);
        p.seek(0.0f); p.play();
        p.tick(5.0f); p.tick(5.0f);
        CHECK(r.ev.empty());
        CHECK(ap(seq.play_head_s, 0.0f));
        CHECK(seq.playing);               // never reached the end
    }
    {   // negative speed → head moves backward; tick() does NOT
        //  lower-clamp (only seek does); no forward events fire.
        cn::Sequence seq;
        seq.duration_s = 10.0f; seq.looping = false; seq.speed = -1.0f;
        seq.tracks.push_back(ev_track({ {1.0f, "n", ""} }));
        cn::Player p(&seq); Rec r; wire(p, r);
        p.seek(5.0f); p.play();
        p.tick(2.0f);                     // t = 5 + 2*(-1) = 3
        CHECK(r.ev.empty());
        CHECK(ap(seq.play_head_s, 3.0f));
    }
}

// ---- camera track shares the identical edge-trigger window ---------
void test_camera_track() {
    cn::Sequence seq;
    seq.duration_s = 10.0f; seq.looping = false; seq.speed = 1.0f;
    cn::CameraTrack ct;
    ct.name = "cam";
    ct.shots.push_back(cn::CameraShot{ "s1", 1.0f, 7u, 0.5f });
    ct.shots.push_back(cn::CameraShot{ "s2", 3.0f, 9u, 0.0f });
    seq.tracks.push_back(ct);
    cn::Player p(&seq); Rec r; wire(p, r);
    p.seek(0.0f); p.play();

    p.tick(2.0f);                         // (0,2] → shot s1
    CHECK(r.cam.size() == sz(1));
    CHECK(r.cam[0] == 7u);
    CHECK(ap(r.blend[0], 0.5f));

    p.tick(2.0f);                         // (2,4] → shot s2 (cut, blend 0)
    CHECK(r.cam.size() == sz(2));
    CHECK(r.cam[1] == 9u);
    CHECK(ap(r.blend[1], 0.0f));

    p.tick(2.0f);                         // (4,6] → nothing; no refire
    CHECK(r.cam.size() == sz(2));
}

// ---- unset callbacks + multi-track ordering ------------------------
void test_callbacks_and_order() {
    {   // No dispatch wired → events are consumed safely (no crash).
        cn::Sequence seq;
        seq.duration_s = 5.0f; seq.looping = false; seq.speed = 1.0f;
        seq.tracks.push_back(ev_track({ {2.0f, "x", ""} }));
        cn::Player p(&seq);               // intentionally NOT wired
        p.seek(0.0f); p.play();
        p.tick(3.0f);
        CHECK(ap(seq.play_head_s, 3.0f)); // advanced fine
        p.tick(3.0f);                     // clamp+stop, still safe
        CHECK(ap(seq.play_head_s, 5.0f));
        CHECK(!seq.playing);
    }
    {   // Two event tracks → both run each tick, in insertion order.
        cn::Sequence seq;
        seq.duration_s = 10.0f; seq.looping = false; seq.speed = 1.0f;
        cn::EventTrack a; a.name = "A"; a.events = { {2.0f, "A", ""} };
        cn::EventTrack b; b.name = "B"; b.events = { {2.0f, "B", ""} };
        seq.tracks.push_back(a);
        seq.tracks.push_back(b);
        cn::Player p(&seq); Rec r; wire(p, r);
        p.seek(0.0f); p.play();
        p.tick(3.0f);                     // both fire @2, track order
        CHECK(r.ev.size() == sz(2));
        CHECK(r.ev[0] == "A");
        CHECK(r.ev[1] == "B");
    }
}

}  // namespace

int main() {
    test_play_state();
    test_seek_clamp();
    test_tick_paused_noop();
    test_event_edge_trigger();
    test_loop_wrap();
    test_non_looping_clamp();
    test_speed();
    test_camera_track();
    test_callbacks_and_order();

    if (g_fail == 0) {
        cardinal::log::infof("cinetest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("cinetest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
