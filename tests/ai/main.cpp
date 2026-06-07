// =============================================================================
// Cardinal — deterministic AI-core regression suite.
//
// Pins the three pure pieces of cardinal::ai:
//
//   Blackboard    — typed variant store. Round-trips every type; a
//                   missing key OR a type-mismatched read returns the
//                   caller's default (the silent footgun); has/clear/
//                   keys()(sorted)/value().
//   Behavior tree — leaves (Wait accumulate+reset, SetBB, Custom,
//                   Inverter, status_name) and the STATEFUL composites:
//                   Sequence/Selector suspend on a Running child and do
//                   NOT re-run earlier children next tick (resume), and
//                   reset their cursor on completion/failure.
//   Perception    — sight cone (radius gate, cos-angle gate with >=
//                   inclusivity, co-located degenerate → dist 0),
//                   hearing (radius × sound strength, <= inclusivity),
//                   stimulus lifetime aging (no event the tick it dies,
//                   lifetime 0 = immortal), deferred remove-compaction,
//                   per-tick event clearing, sensor-major event order,
//                   independent monotonic sensor/stimulus ids.
//
// All geometry is Pythagorean / axis-aligned so expected distances and
// cosines are exact — no <cmath> in the test, no FP-boundary fragility.
// Exit 0 = all pass.
// =============================================================================

#include <cardinal/ai/ai.hpp>
#include <cardinal/core/diag/log.hpp>

#include <string>
#include <utility>          // std::move

namespace {

namespace ai = cardinal::ai;
using Vec3   = cardinal::scene::Vec3;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("aitest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-3f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool streq(const char* a, const char* b) { return std::string(a) == b; }

// Launder a NaN out of pure arithmetic without dragging <cmath>/<limits>
// in (the test file deliberately avoids them — see top-of-file comment).
// The volatile defeats the constant-folder; 0.0/0.0 produces a quiet NaN.
float nan_f() {
    volatile float z = 0.0f;
    return z / z;
}

// ---- Blackboard: round-trip, missing-default, type-mismatch-default -
void test_blackboard() {
    ai::Blackboard bb;

    // Missing key → caller default (every typed accessor).
    CHECK(ap(bb.get_float("nope", 3.5f), 3.5f));
    CHECK(bb.get_int("nope", 7) == 7);
    CHECK(bb.get_bool("nope", true) == true);
    CHECK(bb.get_string("nope", "d") == "d");
    CHECK(ap(bb.get_vec3("nope", Vec3{1, 2, 3}).y, 2.0f));
    CHECK(!bb.has("nope"));
    CHECK(bb.value("nope") == nullptr);

    bb.set_float ("f", 1.25f);
    bb.set_int   ("i", 42);
    bb.set_bool  ("b", true);
    bb.set_vec3  ("v", Vec3{4, 5, 6});
    bb.set_string("s", "hi");

    CHECK(ap(bb.get_float("f"), 1.25f));
    CHECK(bb.get_int("i") == 42);
    CHECK(bb.get_bool("b") == true);
    CHECK(ap(bb.get_vec3("v").z, 6.0f));
    CHECK(bb.get_string("s") == "hi");
    CHECK(bb.has("f") && bb.has("v"));
    CHECK(bb.value("i") != nullptr);

    // Type mismatch → default, NOT a coerced/garbage value. (The footgun:
    // reading a key with the wrong accessor must be safe.)
    CHECK(ap(bb.get_float("i", 9.0f), 9.0f));      // "i" holds int
    CHECK(bb.get_int("f", -1) == -1);              // "f" holds float
    CHECK(bb.get_bool("s", false) == false);       // "s" holds string
    CHECK(bb.get_string("b", "x") == "x");         // "b" holds bool

    // Overwrite (same key, even changing type) is last-write-wins.
    bb.set_float("i", 2.0f);
    CHECK(ap(bb.get_float("i"), 2.0f));
    CHECK(bb.get_int("i", 123) == 123);            // now a float → default

    // keys() is sorted; clear() empties.
    auto ks = bb.keys();
    CHECK(ks.size() == sz(5));
    CHECK(ks.front() == "b" && ks.back() == "v");  // sorted: b,f,i,s,v
    bb.clear();
    CHECK(bb.keys().empty() && !bb.has("f"));
}

// ---- Behavior-tree leaves + Inverter + status_name -----------------
void test_bt_leaves() {
    ai::Blackboard bb;

    CHECK(streq(ai::status_name(ai::Status::Running), "Running"));
    CHECK(streq(ai::status_name(ai::Status::Success), "Success"));
    CHECK(streq(ai::status_name(ai::Status::Failure), "Failure"));

    // Wait: Running until elapsed >= duration, then Success + self-reset
    // (re-usable: a second cycle behaves identically).
    ai::Wait w(0.5f);
    CHECK(w.tick(bb, 0.2f) == ai::Status::Running);
    CHECK(w.tick(bb, 0.2f) == ai::Status::Running);
    CHECK(w.tick(bb, 0.2f) == ai::Status::Success);   // 0.6 >= 0.5
    CHECK(w.tick(bb, 0.2f) == ai::Status::Running);    // reset → counting again
    CHECK(w.tick(bb, 0.4f) == ai::Status::Success);    // 0.6 >= 0.5

    // SetBlackboardFloat: writes key, always Success.
    ai::SetBlackboardFloat sb("hp", 12.0f);
    CHECK(sb.tick(bb, 0.0f) == ai::Status::Success);
    CHECK(ap(bb.get_float("hp"), 12.0f));

    // Custom: forwards lambda result; empty fn → Failure.
    ai::Custom ok("ok", [](ai::Blackboard&, float){ return ai::Status::Success; });
    CHECK(ok.tick(bb, 0.0f) == ai::Status::Success);
    CHECK(streq(ok.name(), "ok"));
    ai::Custom none("none", nullptr);
    CHECK(none.tick(bb, 0.0f) == ai::Status::Failure);

    // Inverter: Success↔Failure, Running passes through, null child → Failure.
    {
        ai::Inverter inv(std::make_unique<ai::Custom>(
            "s", [](ai::Blackboard&, float){ return ai::Status::Success; }));
        CHECK(inv.tick(bb, 0.0f) == ai::Status::Failure);
        CHECK(inv.child_count() == 1u);
    }
    {
        ai::Inverter inv(std::make_unique<ai::Custom>(
            "f", [](ai::Blackboard&, float){ return ai::Status::Failure; }));
        CHECK(inv.tick(bb, 0.0f) == ai::Status::Success);
    }
    {
        ai::Inverter inv(std::make_unique<ai::Custom>(
            "r", [](ai::Blackboard&, float){ return ai::Status::Running; }));
        CHECK(inv.tick(bb, 0.0f) == ai::Status::Running);
    }
    {
        ai::Inverter inv(nullptr);
        CHECK(inv.tick(bb, 0.0f) == ai::Status::Failure);
        CHECK(inv.child_count() == 0u);
    }
}

// ---- Sequence: all-success / fail-short-circuit / Running resume ---
void test_bt_sequence() {
    ai::Blackboard bb;

    // Empty Sequence → Success.
    { ai::Sequence s; CHECK(s.tick(bb, 0.0f) == ai::Status::Success); }

    // All children Success → Success; child_count/child() introspection.
    {
        ai::Sequence s;
        s.add(std::make_unique<ai::Custom>("a",
              [](ai::Blackboard&, float){ return ai::Status::Success; }));
        s.add(std::make_unique<ai::Custom>("b",
              [](ai::Blackboard&, float){ return ai::Status::Success; }));
        CHECK(s.child_count() == 2u);
        CHECK(s.child(0) != nullptr && s.child(5) == nullptr);
        CHECK(s.tick(bb, 0.0f) == ai::Status::Success);
    }

    // First failure short-circuits: later children never tick, and the
    // cursor resets so a re-tick restarts from the front.
    {
        int a_runs = 0, c_runs = 0;
        ai::Sequence s;
        s.add(std::make_unique<ai::Custom>("a",
              [&](ai::Blackboard&, float){ ++a_runs; return ai::Status::Success; }));
        s.add(std::make_unique<ai::Custom>("b",
              [](ai::Blackboard&, float){ return ai::Status::Failure; }));
        s.add(std::make_unique<ai::Custom>("c",
              [&](ai::Blackboard&, float){ ++c_runs; return ai::Status::Success; }));
        CHECK(s.tick(bb, 0.0f) == ai::Status::Failure);
        CHECK(s.tick(bb, 0.0f) == ai::Status::Failure);
        CHECK(a_runs == 2);          // restarted from front each tick
        CHECK(c_runs == 0);          // never reached past the failure
    }

    // Running child suspends the sequence; earlier children must NOT
    // re-run on the next tick (the stateful-resume contract).
    {
        int set_runs = 0;
        ai::Sequence s;
        s.add(std::make_unique<ai::Custom>("set",
              [&](ai::Blackboard& b, float){ ++set_runs;
                                             b.set_float("k", 7.0f);
                                             return ai::Status::Success; }));
        s.add(std::make_unique<ai::Wait>(0.5f));
        CHECK(s.tick(bb, 0.2f) == ai::Status::Running);   // set ran, Wait Running
        CHECK(set_runs == 1);
        bb.set_float("k", 0.0f);                          // sentinel
        CHECK(s.tick(bb, 0.2f) == ai::Status::Running);   // resume at Wait only
        CHECK(s.tick(bb, 0.2f) == ai::Status::Success);   // Wait 0.6 >= 0.5
        CHECK(set_runs == 1);                             // NEVER re-ran
        CHECK(ap(bb.get_float("k"), 0.0f));               // not re-set to 7
    }
}

// ---- Selector: first-success / all-fail / Running resume -----------
void test_bt_selector() {
    ai::Blackboard bb;

    { ai::Selector s; CHECK(s.tick(bb, 0.0f) == ai::Status::Failure); }  // empty

    // First Success wins; later children never tick.
    {
        int b_runs = 0;
        ai::Selector s;
        s.add(std::make_unique<ai::Custom>("a",
              [](ai::Blackboard&, float){ return ai::Status::Success; }));
        s.add(std::make_unique<ai::Custom>("b",
              [&](ai::Blackboard&, float){ ++b_runs; return ai::Status::Success; }));
        CHECK(s.tick(bb, 0.0f) == ai::Status::Success);
        CHECK(b_runs == 0);
    }

    // All Failure → Failure (cursor resets → re-tick restarts at front).
    {
        int a_runs = 0;
        ai::Selector s;
        s.add(std::make_unique<ai::Custom>("a",
              [&](ai::Blackboard&, float){ ++a_runs; return ai::Status::Failure; }));
        s.add(std::make_unique<ai::Custom>("b",
              [](ai::Blackboard&, float){ return ai::Status::Failure; }));
        CHECK(s.tick(bb, 0.0f) == ai::Status::Failure);
        CHECK(s.tick(bb, 0.0f) == ai::Status::Failure);
        CHECK(a_runs == 2);
    }

    // A failing child then a Running child: selector returns Running and
    // resumes at the Running child — the failed one does not re-run.
    {
        int fail_runs = 0;
        ai::Selector s;
        s.add(std::make_unique<ai::Custom>("fail",
              [&](ai::Blackboard&, float){ ++fail_runs; return ai::Status::Failure; }));
        s.add(std::make_unique<ai::Wait>(0.5f));
        CHECK(s.tick(bb, 0.2f) == ai::Status::Running);
        CHECK(fail_runs == 1);
        CHECK(s.tick(bb, 0.2f) == ai::Status::Running);   // resume at Wait
        CHECK(fail_runs == 1);                            // not re-run
        CHECK(s.tick(bb, 0.2f) == ai::Status::Success);   // Wait completes
    }
}

// ---- Perception: sight cone (radius / angle / degenerate) ----------
void test_perception_sight() {
    ai::PerceptionWorld w;

    ai::SensorDesc s{};
    s.position = Vec3{0, 0, 0};
    s.forward  = Vec3{0, 0, -1};       // unit → cos_a is a true cosine
    s.sight_radius    = 25.0f;
    s.sight_cos_angle = 0.5f;          // ~60° half-FOV
    s.hearing_enabled = false;
    const ai::SensorId sid = w.add_sensor(s);
    CHECK(sid == 1u);

    auto sight_at = [&](float x, float y, float z) {
        ai::StimulusDesc d{};
        d.kind = ai::StimulusKind::Sight;
        d.position = Vec3{x, y, z};
        d.lifetime_seconds = 0.0f;     // immortal — geometry only
        return w.add_stimulus(d);
    };

    // Dead-ahead, in range: event, exact distance, cos_a = 1.0.
    const ai::StimulusId t1 = sight_at(0.0f, 0.0f, -10.0f);
    w.tick(0.016f);
    CHECK(w.last_events().size() == sz(1));
    {
        const ai::PerceptionEvent& e = w.last_events()[0];
        CHECK(e.sensor == sid && e.stimulus == t1);
        CHECK(e.kind == ai::StimulusKind::Sight);
        CHECK(ap(e.distance, 10.0f));
        CHECK(ap(e.position.z, -10.0f));
    }
    w.remove_stimulus(t1); w.tick(0.0f);   // clear it out

    // 90° to the side → cos_a 0 < 0.5 → not seen (in range, wrong angle).
    const ai::StimulusId t2 = sight_at(10.0f, 0.0f, 0.0f);
    w.tick(0.016f);
    CHECK(w.last_events().empty());
    w.remove_stimulus(t2); w.tick(0.0f);

    // (3,0,-4)·k direction → cos_a = 0.8. Inside a 0.5 gate, outside 0.9.
    const ai::StimulusId t3 = sight_at(3.0f, 0.0f, -4.0f);  // dist 5, cos 0.8
    w.tick(0.016f);
    CHECK(w.last_events().size() == sz(1));
    CHECK(ap(w.last_events()[0].distance, 5.0f));
    {
        ai::SensorDesc tight = s; tight.sight_cos_angle = 0.9f;
        w.update_sensor(sid, tight);
        w.tick(0.016f);
        CHECK(w.last_events().empty());              // 0.8 < 0.9
        w.update_sensor(sid, s);                     // restore 0.5 gate
    }
    w.remove_stimulus(t3); w.tick(0.0f);

    // In cone but beyond sight_radius → rejected by the range gate.
    const ai::StimulusId t4 = sight_at(0.0f, 0.0f, -30.0f);  // dist 30 > 25
    w.tick(0.016f);
    CHECK(w.last_events().empty());
    w.remove_stimulus(t4); w.tick(0.0f);

    // `>=` inclusivity: cos_a exactly 1.0, gate exactly 1.0 → seen.
    {
        ai::SensorDesc g = s; g.sight_cos_angle = 1.0f;
        w.update_sensor(sid, g);
        const ai::StimulusId t = sight_at(0.0f, 0.0f, -7.0f);  // straight ahead
        w.tick(0.016f);
        CHECK(w.last_events().size() == sz(1));       // 1.0 >= 1.0
        w.remove_stimulus(t); w.tick(0.0f);
        w.update_sensor(sid, s);
    }

    // Co-located → degenerate branch: event with distance exactly 0,
    // regardless of facing.
    {
        ai::SensorDesc back = s; back.forward = Vec3{0, 0, 1}; // facing away
        w.update_sensor(sid, back);
        const ai::StimulusId t = sight_at(0.0f, 0.0f, 0.0f);  // sensor point
        w.tick(0.016f);
        CHECK(w.last_events().size() == sz(1));
        CHECK(ap(w.last_events()[0].distance, 0.0f, 1e-6f));
        w.remove_stimulus(t); w.tick(0.0f);
        w.update_sensor(sid, s);
    }

    // sight_enabled = false → blind even to a dead-ahead target.
    {
        ai::SensorDesc off = s; off.sight_enabled = false;
        w.update_sensor(sid, off);
        const ai::StimulusId t = sight_at(0.0f, 0.0f, -5.0f);
        w.tick(0.016f);
        CHECK(w.last_events().empty());
        w.remove_stimulus(t); w.tick(0.0f);
        w.update_sensor(sid, s);
    }
}

// ---- Perception: hearing (radius × strength, <= inclusivity) -------
void test_perception_hearing() {
    ai::PerceptionWorld w;

    ai::SensorDesc s{};
    s.position = Vec3{0, 0, 0};
    s.sight_enabled  = false;
    s.hearing_enabled = true;
    s.hearing_radius  = 10.0f;
    const ai::SensorId sid = w.add_sensor(s);

    auto sound_at = [&](float z, float strength) {
        ai::StimulusDesc d{};
        d.kind = ai::StimulusKind::Sound;
        d.position = Vec3{0, 0, z};
        d.strength = strength;
        d.lifetime_seconds = 0.0f;
        return w.add_stimulus(d);
    };

    // Inside radius (strength 1 → effective 10): dist 8 → heard.
    const ai::StimulusId a = sound_at(-8.0f, 1.0f);
    w.tick(0.016f);
    CHECK(w.last_events().size() == sz(1));
    CHECK(w.last_events()[0].kind == ai::StimulusKind::Sound);
    CHECK(ap(w.last_events()[0].distance, 8.0f));
    w.remove_stimulus(a); w.tick(0.0f);

    // `<=` inclusivity: dist exactly 10 == effective 10 → heard.
    const ai::StimulusId b = sound_at(-10.0f, 1.0f);
    w.tick(0.016f);
    CHECK(w.last_events().size() == sz(1));
    w.remove_stimulus(b); w.tick(0.0f);

    // Just outside: dist 11 > 10 → not heard.
    const ai::StimulusId c = sound_at(-11.0f, 1.0f);
    w.tick(0.016f);
    CHECK(w.last_events().empty());
    w.remove_stimulus(c); w.tick(0.0f);

    // Strength scales the radius. dist 15: inaudible at ×1 (eff 10),
    // audible at ×2 (eff 20).
    {
        const ai::StimulusId q = sound_at(-15.0f, 1.0f);
        w.tick(0.016f);
        CHECK(w.last_events().empty());
        w.remove_stimulus(q); w.tick(0.0f);
    }
    {
        const ai::StimulusId q = sound_at(-15.0f, 2.0f);
        w.tick(0.016f);
        CHECK(w.last_events().size() == sz(1));
        w.remove_stimulus(q); w.tick(0.0f);
    }
    // Strength 0.5 shrinks it: dist 6 vs effective 5 → inaudible.
    {
        const ai::StimulusId q = sound_at(-6.0f, 0.5f);
        w.tick(0.016f);
        CHECK(w.last_events().empty());
        w.remove_stimulus(q); w.tick(0.0f);
    }

    // hearing_enabled = false → deaf even point-blank.
    {
        ai::SensorDesc deaf = s; deaf.hearing_enabled = false;
        w.update_sensor(sid, deaf);
        const ai::StimulusId q = sound_at(-1.0f, 1.0f);
        w.tick(0.016f);
        CHECK(w.last_events().empty());
        w.remove_stimulus(q); w.tick(0.0f);
    }
}

// ---- Perception: lifetime, deferred compaction, order, ids ---------
void test_perception_lifecycle() {
    // Independent monotonic id counters; ids are NOT reused after remove.
    {
        ai::PerceptionWorld w;
        ai::SensorDesc sd{}; sd.position = Vec3{0,0,0};
        ai::StimulusDesc td{}; td.kind = ai::StimulusKind::Sight;
        const ai::SensorId  s1 = w.add_sensor(sd);
        const ai::SensorId  s2 = w.add_sensor(sd);
        const ai::StimulusId x1 = w.add_stimulus(td);
        const ai::StimulusId x2 = w.add_stimulus(td);
        CHECK(s1 == 1u && s2 == 2u);          // sensor counter
        CHECK(x1 == 1u && x2 == 2u);          // independent stimulus counter
        w.remove_sensor(s1);
        w.tick(0.0f);                         // compacts s1
        CHECK(w.sensor_count() == sz(1));
        const ai::SensorId s3 = w.add_sensor(sd);
        CHECK(s3 == 3u);                      // not reused (was 1)
    }

    // Stimulus lifetime: heard while young; the tick that ages it past
    // its lifetime produces NO event and removes it (age happens before
    // the event pass).
    {
        ai::PerceptionWorld w;
        ai::SensorDesc s{}; s.position = Vec3{0,0,0};
        s.forward = Vec3{0,0,-1}; s.sight_radius = 50.0f;
        s.sight_cos_angle = 0.0f; s.hearing_enabled = false;
        w.add_sensor(s);
        ai::StimulusDesc d{};
        d.kind = ai::StimulusKind::Sight;
        d.position = Vec3{0,0,-5};
        d.lifetime_seconds = 0.5f;
        w.add_stimulus(d);

        w.tick(0.4f);                          // age 0.4 < 0.5 → alive
        CHECK(w.last_events().size() == sz(1));
        CHECK(w.stimulus_count() == sz(1));
        w.tick(0.4f);                          // age 0.8 >= 0.5 → dies first
        CHECK(w.last_events().empty());        // no event the tick it dies
        CHECK(w.stimulus_count() == sz(0));    // compacted out
        w.tick(0.4f);
        CHECK(w.last_events().empty());        // stays gone
    }

    // lifetime_seconds == 0 → immortal: still firing after long elapsed.
    {
        ai::PerceptionWorld w;
        ai::SensorDesc s{}; s.position = Vec3{0,0,0};
        s.forward = Vec3{0,0,-1}; s.sight_cos_angle = 0.0f;
        s.sight_radius = 50.0f; s.hearing_enabled = false;
        w.add_sensor(s);
        ai::StimulusDesc d{};
        d.kind = ai::StimulusKind::Sight; d.position = Vec3{0,0,-5};
        d.lifetime_seconds = 0.0f;
        w.add_stimulus(d);
        for (int i = 0; i < 100; ++i) w.tick(1.0f);
        CHECK(w.last_events().size() == sz(1));
        CHECK(w.stimulus_count() == sz(1));
    }

    // remove_* is deferred: count holds until the next tick compacts,
    // and the removed stimulus fires no event during that tick.
    {
        ai::PerceptionWorld w;
        ai::SensorDesc s{}; s.position = Vec3{0,0,0};
        s.forward = Vec3{0,0,-1}; s.sight_cos_angle = 0.0f;
        s.sight_radius = 50.0f; s.hearing_enabled = false;
        w.add_sensor(s);
        ai::StimulusDesc d{};
        d.kind = ai::StimulusKind::Sight; d.position = Vec3{0,0,-5};
        d.lifetime_seconds = 0.0f;
        const ai::StimulusId id = w.add_stimulus(d);
        w.tick(0.016f);
        CHECK(w.last_events().size() == sz(1));
        w.remove_stimulus(id);
        CHECK(w.stimulus_count() == sz(1));    // not compacted yet
        w.tick(0.016f);
        CHECK(w.last_events().empty());        // dead → skipped this tick
        CHECK(w.stimulus_count() == sz(0));    // now compacted
    }

    // Two sensors × two stimuli, all visible → 4 events in sensor-major,
    // stimulus-minor insertion order; last_events_ cleared each tick.
    {
        ai::PerceptionWorld w;
        ai::SensorDesc s{}; s.position = Vec3{0,0,0};
        s.forward = Vec3{0,0,-1}; s.sight_cos_angle = 0.0f;
        s.sight_radius = 50.0f; s.hearing_enabled = false;
        const ai::SensorId s1 = w.add_sensor(s);
        const ai::SensorId s2 = w.add_sensor(s);
        ai::StimulusDesc d{}; d.kind = ai::StimulusKind::Sight;
        d.lifetime_seconds = 0.0f;
        d.position = Vec3{0,0,-5};  const ai::StimulusId x1 = w.add_stimulus(d);
        d.position = Vec3{0,0,-9};  const ai::StimulusId x2 = w.add_stimulus(d);

        w.tick(0.016f);
        const auto& ev = w.last_events();
        CHECK(ev.size() == sz(4));
        CHECK(ev[0].sensor == s1 && ev[0].stimulus == x1);
        CHECK(ev[1].sensor == s1 && ev[1].stimulus == x2);
        CHECK(ev[2].sensor == s2 && ev[2].stimulus == x1);
        CHECK(ev[3].sensor == s2 && ev[3].stimulus == x2);
        CHECK(ap(ev[1].distance, 9.0f) && ap(ev[2].distance, 5.0f));

        w.remove_stimulus(x1);
        w.remove_stimulus(x2);
        w.tick(0.016f);
        CHECK(w.last_events().empty());        // per-tick clear + all gone
    }
}

// ---- Non-finite dt must NOT poison the tick accumulators -----------
// Same accumulator-poison shape that bit physics/sim/cine/particles/
// audio/anim — `accum += NaN` makes accum NaN, then `accum >= threshold`
// is false for NaN → the gate never fires and the state is stuck. Here
// it would hang Wait forever (behavior tree never advances) and make
// PerceptionWorld stimuli immortal (memory leak + permanent fake
// sight/hearing events).
void test_nonfinite_dt() {
    ai::Blackboard bb;

    // Wait: a NaN frame must skip the accumulator and stay Running;
    // subsequent finite ticks must still progress and reach Success.
    ai::Wait w(0.5f);
    CHECK(w.tick(bb, 0.2f)       == ai::Status::Running);
    CHECK(w.tick(bb, nan_f())    == ai::Status::Running);   // skip, elapsed_ still 0.2
    CHECK(w.tick(bb, 0.4f)       == ai::Status::Success);   // 0.2+0.4 = 0.6 >= 0.5

    // PerceptionWorld: NaN dt clears events (no stale leak) and skips
    // aging; stimulus stays alive at its prior age so the NEXT finite
    // tick crosses the lifetime threshold on schedule.
    ai::PerceptionWorld pw;
    ai::SensorDesc s{};
    s.position        = Vec3{0, 0, 0};
    s.forward         = Vec3{0, 0, -1};
    s.sight_radius    = 50.0f;
    s.sight_cos_angle = 0.0f;
    s.hearing_enabled = false;
    pw.add_sensor(s);
    ai::StimulusDesc d{};
    d.kind             = ai::StimulusKind::Sight;
    d.position         = Vec3{0, 0, -5};
    d.lifetime_seconds = 0.5f;
    pw.add_stimulus(d);

    pw.tick(0.2f);                              // age 0.2 < 0.5 → alive, event fires
    CHECK(pw.last_events().size() == sz(1));
    pw.tick(nan_f());                            // NaN → events cleared, age unchanged
    CHECK(pw.last_events().empty());
    CHECK(pw.stimulus_count()    == sz(1));     // still alive (no NaN poison)
    pw.tick(0.4f);                              // 0.2+0.4 = 0.6 >= 0.5 → dies, no event
    CHECK(pw.last_events().empty());
    CHECK(pw.stimulus_count()    == sz(0));     // compacted out
}

}  // namespace

int main() {
    test_blackboard();
    test_bt_leaves();
    test_bt_sequence();
    test_bt_selector();
    test_perception_sight();
    test_perception_hearing();
    test_perception_lifecycle();
    test_nonfinite_dt();

    if (g_fail == 0) {
        cardinal::log::infof("aitest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("aitest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
