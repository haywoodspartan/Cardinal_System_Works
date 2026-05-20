// =============================================================================
// Cardinal — deterministic input-manager regression suite.
//
// input::Manager is a pure state machine: producers call push_*() (the
// exact API the OS hook uses) and begin_frame(dt) drains them, recomputes
// rising/falling edges, and decays mouse delta/wheel. Gameplay relies on
// a precise contract this suite pins:
//
//   * pressed / released are ONE-frame edge pulses;
//   * held_time accumulates in the pre-pass while down (INCLUDING the
//     release frame) and resets to 0 ONLY on a fresh press — never on
//     release, so it goes stale by design;
//   * an action is the OR of its bindings; an axis is the clamped
//     ([-1,1]) sum of scaled contributions; gamepad bindings are inert;
//   * the PIE gate zeroes action_*/axis but NOT direct key/mouse
//     queries (Studio hotkeys must survive a paused game);
//   * events queued before begin_frame are invisible until it drains;
//   * mouse delta/wheel are per-frame (the x<0 sentinel quirk included).
//
// The Win32 hook (input_windows.cpp) is OS-bound and out of scope; the
// producer-side push_* path drives the identical state. Pure, headless,
// fully deterministic. Exit 0 = all pass.
// =============================================================================

#include <cardinal/input/input.hpp>
#include <cardinal/core/log.hpp>

#include <string>
#include <vector>

namespace {

namespace in = cardinal::input;
using in::KeyCode;
using in::MouseButton;

int g_checks = 0;
int g_fail   = 0;

void check_impl(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        cardinal::log::errorf("inputtest", "FAIL  L%d  %s", line, expr);
    }
}
#define CHECK(x) ::check_impl(static_cast<bool>(x), #x, __LINE__)

bool ap(float a, float b, float e = 1e-4f) {
    const float d = (a > b) ? (a - b) : (b - a);
    return d <= e;
}
cardinal::usize sz(int n) { return static_cast<cardinal::usize>(n); }
bool streq(const char* a, const char* b) { return std::string(a) == b; }

// down / pressed / released triad compare (held_time checked separately).
bool bs(const in::ButtonState& s, bool d, bool p, bool r) {
    return s.down == d && s.pressed == p && s.released == r;
}
// Launder a NaN without dragging <cmath>/<limits> in. volatile defeats
// the constant-folder so 0.0/0.0 actually executes and yields qNaN.
float nan_f() { volatile float z = 0.0f; return z / z; }
in::Binding kb(KeyCode k, float scale = 1.0f) {
    return in::Binding{ in::BindKind::Key,
                        static_cast<cardinal::u32>(k), scale };
}
in::Binding mbb(MouseButton b, float scale = 1.0f) {
    return in::Binding{ in::BindKind::MouseButton,
                        static_cast<cardinal::u32>(b), scale };
}

// ---- key_code_name / key_code_from_string round-trip --------------
void test_keycode_strings() {
    CHECK(streq(in::key_code_name(KeyCode::A), "A"));
    CHECK(streq(in::key_code_name(KeyCode::Z), "Z"));
    CHECK(streq(in::key_code_name(KeyCode::K0), "0"));
    CHECK(streq(in::key_code_name(KeyCode::Space), "Space"));
    CHECK(streq(in::key_code_name(KeyCode::LeftShift), "LShift"));
    CHECK(streq(in::key_code_name(KeyCode::F12), "F12"));
    // Unnamed codes + sentinels fall through to "Unknown".
    CHECK(streq(in::key_code_name(KeyCode::Unknown), "Unknown"));
    CHECK(streq(in::key_code_name(KeyCode::Count), "Unknown"));
    CHECK(streq(in::key_code_name(KeyCode::NumPad0), "Unknown"));

    // Single-char parse: letters (both cases) + digits.
    CHECK(in::key_code_from_string("A") == KeyCode::A);
    CHECK(in::key_code_from_string("a") == KeyCode::A);
    CHECK(in::key_code_from_string("z") == KeyCode::Z);
    CHECK(in::key_code_from_string("0") == KeyCode::K0);
    CHECK(in::key_code_from_string("9") == KeyCode::K9);
    // Multi-char names via the table.
    CHECK(in::key_code_from_string("Space") == KeyCode::Space);
    CHECK(in::key_code_from_string("F5")    == KeyCode::F5);
    CHECK(in::key_code_from_string("LShift")== KeyCode::LeftShift);
    CHECK(in::key_code_from_string("Up")    == KeyCode::Up);
    // Robust to junk.
    CHECK(in::key_code_from_string(nullptr) == KeyCode::Unknown);
    CHECK(in::key_code_from_string("")      == KeyCode::Unknown);
    CHECK(in::key_code_from_string("nope")  == KeyCode::Unknown);

    // Exact round-trip across every NAMED code (unique names).
    bool rt = true;
    for (cardinal::u32 i = static_cast<cardinal::u32>(KeyCode::A);
         i <= static_cast<cardinal::u32>(KeyCode::Z); ++i) {
        const auto k = static_cast<KeyCode>(i);
        if (in::key_code_from_string(in::key_code_name(k)) != k) rt = false;
    }
    for (cardinal::u32 i = static_cast<cardinal::u32>(KeyCode::F1);
         i <= static_cast<cardinal::u32>(KeyCode::F12); ++i) {
        const auto k = static_cast<KeyCode>(i);
        if (in::key_code_from_string(in::key_code_name(k)) != k) rt = false;
    }
    CHECK(rt);
    CHECK(in::key_code_from_string(in::key_code_name(KeyCode::Escape))
          == KeyCode::Escape);
    CHECK(in::key_code_from_string(in::key_code_name(KeyCode::RightCtrl))
          == KeyCode::RightCtrl);
}

// ---- edge lifecycle: press / hold / release / stale / re-press ----
void test_key_edges() {
    auto mp = in::Manager::create();
    auto& M = *mp;
    const auto& W = M.key(KeyCode::W);

    CHECK(!M.key_down(KeyCode::W));                 // pristine

    // F1: rising edge.
    M.push_key(KeyCode::W, true);  M.begin_frame(0.5f);
    CHECK(bs(W, true, true, false));  CHECK(ap(W.held_time, 0.0f));

    // F2: still held — pressed clears, held_time accrues in the pre-pass.
    M.begin_frame(0.5f);
    CHECK(bs(W, true, false, false)); CHECK(ap(W.held_time, 0.5f));

    // F3: keeps accruing.
    M.begin_frame(0.5f);
    CHECK(bs(W, true, false, false)); CHECK(ap(W.held_time, 1.0f));

    // F4: falling edge — release frame STILL accrues held_time in the
    // pre-pass before the up event is drained.
    M.push_key(KeyCode::W, false); M.begin_frame(0.5f);
    CHECK(bs(W, false, false, true)); CHECK(ap(W.held_time, 1.5f));

    // F5: released is a one-frame pulse; held_time goes stale (NOT reset).
    M.begin_frame(0.5f);
    CHECK(bs(W, false, false, false)); CHECK(ap(W.held_time, 1.5f));

    // F6: fresh press resets held_time to 0.
    M.push_key(KeyCode::W, true);  M.begin_frame(0.5f);
    CHECK(bs(W, true, true, false));  CHECK(ap(W.held_time, 0.0f));
}

// ---- taps, redundant events, query-before-frame -------------------
void test_key_taps() {
    {   // press + release in ONE frame → both pulses, held 0.
        auto mp = in::Manager::create(); auto& M = *mp;
        M.push_key(KeyCode::J, true);
        M.push_key(KeyCode::J, false);
        M.begin_frame(0.5f);
        const auto& J = M.key(KeyCode::J);
        CHECK(bs(J, false, true, true)); CHECK(ap(J.held_time, 0.0f));
    }
    {   // double "down" same frame is idempotent (one press, stays down).
        auto mp = in::Manager::create(); auto& M = *mp;
        M.push_key(KeyCode::K, true);
        M.push_key(KeyCode::K, true);
        M.begin_frame(0.5f);
        CHECK(bs(M.key(KeyCode::K), true, true, false));
    }
    {   // redundant "down" on a later frame → no spurious re-press,
        //  held_time keeps accruing (not reset).
        auto mp = in::Manager::create(); auto& M = *mp;
        M.push_key(KeyCode::L, true); M.begin_frame(0.5f);
        CHECK(bs(M.key(KeyCode::L), true, true, false));
        M.push_key(KeyCode::L, true); M.begin_frame(0.5f);
        CHECK(bs(M.key(KeyCode::L), true, false, false));
        CHECK(ap(M.key(KeyCode::L).held_time, 0.5f));
    }
    {   // events are invisible until begin_frame drains them.
        auto mp = in::Manager::create(); auto& M = *mp;
        M.push_key(KeyCode::A, true);
        CHECK(!M.key_down(KeyCode::A));             // not drained yet
        M.begin_frame(0.5f);
        CHECK(M.key_down(KeyCode::A));
        CHECK(M.key_pressed(KeyCode::A));
    }
}

// ---- out-of-range key / mouse handles are inert -------------------
void test_bounds() {
    auto mp = in::Manager::create(); auto& M = *mp;
    CHECK(!M.key(KeyCode::Unknown).down);
    CHECK(!M.key_down(KeyCode::Unknown));
    CHECK(!M.key(KeyCode::Count).down);
    CHECK(!M.mouse(MouseButton::Count).down);
    // Pushing Unknown/out-of-range is silently dropped (no crash).
    M.push_key(KeyCode::Unknown, true);
    M.begin_frame(0.5f);
    CHECK(!M.key_down(KeyCode::Unknown));
}

// ---- action map: OR of bindings, gamepad inert, clear/names -------
void test_actions() {
    auto mp = in::Manager::create(); auto& M = *mp;
    M.bind_action("Fire", kb(KeyCode::F));
    M.bind_action("Fire", mbb(MouseButton::Left));
    M.bind_action("Pad",  in::Binding{ in::BindKind::GamepadButton,
        static_cast<cardinal::u32>(in::GamepadButton::A), 1.0f });

    CHECK(!M.action_down("Fire"));                  // nothing pressed
    CHECK(!M.action_down("Missing"));               // unknown action

    M.push_key(KeyCode::F, true); M.begin_frame(0.5f);
    CHECK(M.action_down("Fire"));
    CHECK(M.action_pressed("Fire"));
    CHECK(!M.action_released("Fire"));

    M.begin_frame(0.5f);                            // hold
    CHECK(M.action_down("Fire"));
    CHECK(!M.action_pressed("Fire"));

    M.push_key(KeyCode::F, false); M.begin_frame(0.5f);
    CHECK(!M.action_down("Fire"));
    CHECK(M.action_released("Fire"));

    // OR semantics: the mouse binding alone satisfies the action.
    M.push_mouse_button(MouseButton::Left, true); M.begin_frame(0.5f);
    CHECK(M.action_down("Fire"));

    // Gamepad bindings are inert (not yet wired) — never down.
    CHECK(!M.action_down("Pad"));

    // Inspection + clear.
    M.bind_action("Zeta", kb(KeyCode::Z));
    M.bind_action("Alpha", kb(KeyCode::A));
    auto names = M.action_names();
    CHECK(names.size() == sz(4));
    CHECK(names.front() == "Alpha" && names.back() == "Zeta");  // sorted
    CHECK(M.action_bindings("Fire").size() == sz(2));
    CHECK(M.action_bindings("Missing").empty());    // stable empty ref
    M.clear_action("Fire");
    CHECK(!M.action_down("Fire"));
    CHECK(M.action_bindings("Fire").empty());
}

// ---- axis: clamped sum of scaled contributions --------------------
void test_axis() {
    auto mp = in::Manager::create(); auto& M = *mp;
    M.bind_axis("MoveX", kb(KeyCode::D,  1.0f));
    M.bind_axis("MoveX", kb(KeyCode::A, -1.0f));

    CHECK(ap(M.axis("MoveX"), 0.0f));               // nothing held
    CHECK(ap(M.axis("Missing"), 0.0f));             // unknown axis

    M.push_key(KeyCode::D, true); M.begin_frame(0.5f);
    CHECK(ap(M.axis("MoveX"), 1.0f));               // +1

    M.push_key(KeyCode::A, true); M.begin_frame(0.5f);
    CHECK(ap(M.axis("MoveX"), 0.0f));               // +1 -1 cancel

    M.push_key(KeyCode::D, false); M.begin_frame(0.5f);
    CHECK(ap(M.axis("MoveX"), -1.0f));              // only -1 left

    // Clamp: two like-signed contributions saturate at ±1.
    M.bind_axis("Boost", kb(KeyCode::W, 1.0f));
    M.bind_axis("Boost", kb(KeyCode::E, 1.0f));
    M.push_key(KeyCode::W, true);
    M.push_key(KeyCode::E, true);
    M.begin_frame(0.5f);
    CHECK(ap(M.axis("Boost"), 1.0f));               // 2 → clamp +1

    M.bind_axis("Neg", kb(KeyCode::X, -1.0f));
    M.bind_axis("Neg", kb(KeyCode::Y, -1.0f));
    M.push_key(KeyCode::X, true);
    M.push_key(KeyCode::Y, true);
    M.begin_frame(0.5f);
    CHECK(ap(M.axis("Neg"), -1.0f));                // -2 → clamp -1

    auto an = M.axis_names();
    CHECK(an.size() == sz(3));                             // MoveX/Boost/Neg
    CHECK(an.front() == "Boost" && an.back() == "Neg");   // sorted
    M.clear_axis("MoveX");
    CHECK(ap(M.axis("MoveX"), 0.0f));
    CHECK(M.axis_bindings("MoveX").empty());

    M.reset_bindings();
    CHECK(M.action_names().empty() && M.axis_names().empty());
}

// ---- PIE gameplay gate: gates actions/axes, not direct queries ----
void test_gameplay_gate() {
    auto mp = in::Manager::create(); auto& M = *mp;
    M.bind_action("Jump", kb(KeyCode::Space));
    M.bind_axis("Mv", kb(KeyCode::D, 1.0f));

    M.push_key(KeyCode::Space, true);
    M.push_key(KeyCode::D, true);
    M.begin_frame(0.5f);

    CHECK(M.is_gameplay_active());                  // default on
    CHECK(M.action_down("Jump"));
    CHECK(M.action_pressed("Jump"));
    CHECK(ap(M.axis("Mv"), 1.0f));
    CHECK(M.key_down(KeyCode::Space));

    M.set_gameplay_active(false);
    CHECK(!M.is_gameplay_active());
    CHECK(!M.action_down("Jump"));                  // gated off
    CHECK(!M.action_pressed("Jump"));
    CHECK(!M.action_released("Jump"));
    CHECK(ap(M.axis("Mv"), 0.0f));                  // gated off
    CHECK(M.key_down(KeyCode::Space));              // NOT gated
    CHECK(M.key(KeyCode::Space).down);              // direct query survives

    M.set_gameplay_active(true);
    CHECK(M.action_down("Jump"));                   // restored
    CHECK(ap(M.axis("Mv"), 1.0f));
}

// ---- mouse: per-frame delta + wheel + sentinel + button edges -----
void test_mouse() {
    auto mp = in::Manager::create(); auto& M = *mp;
    const auto& ms = M.mouse_state();
    CHECK(ms.x == 0 && ms.y == 0 && ms.dx == 0 && ms.dy == 0);

    M.push_mouse_move(10, 5); M.begin_frame(0.016f);
    CHECK(ms.x == 10 && ms.y == 5 && ms.dx == 10 && ms.dy == 5);

    M.begin_frame(0.016f);                          // no move → delta 0
    CHECK(ms.x == 10 && ms.y == 5 && ms.dx == 0 && ms.dy == 0);

    M.push_mouse_move(13, 9); M.begin_frame(0.016f);
    CHECK(ms.x == 13 && ms.dx == 3 && ms.dy == 4);

    M.push_mouse_move(100, 100);                    // superseded...
    M.push_mouse_move(20, 20);                      // ...last write wins
    M.begin_frame(0.016f);
    CHECK(ms.x == 20 && ms.y == 20 && ms.dx == 7 && ms.dy == 11);

    // x < 0 collides with the "no pending move" sentinel → swallowed.
    M.push_mouse_move(-1, 50); M.begin_frame(0.016f);
    CHECK(ms.x == 20 && ms.y == 20 && ms.dx == 0 && ms.dy == 0);

    // Wheel accumulates within a frame, resets the next.
    M.push_mouse_wheel(3);
    M.push_mouse_wheel(-1);
    M.begin_frame(0.016f);
    CHECK(ms.wheel == 2);
    M.begin_frame(0.016f);
    CHECK(ms.wheel == 0);

    // Mouse-button edges mirror keys.
    M.push_mouse_button(MouseButton::Left, true); M.begin_frame(0.016f);
    CHECK(bs(M.mouse(MouseButton::Left), true, true, false));
    CHECK(M.mouse_pressed(MouseButton::Left));
    M.begin_frame(0.016f);
    CHECK(bs(M.mouse(MouseButton::Left), true, false, false));
    M.push_mouse_button(MouseButton::Left, false); M.begin_frame(0.016f);
    CHECK(bs(M.mouse(MouseButton::Left), false, false, true));
    CHECK(!M.mouse_down(MouseButton::Left));
}

// ---- non-finite dt must NOT poison held_time, must STILL drain events
// `held_time += NaN` makes it NaN, and the typical `held_time >= thresh`
// auto-repeat / hold-to-action gate stays false forever (NaN compares
// unordered) — the poison persists until the user releases AND re-
// presses the key, since fresh-down is the only reset. Event drain
// must continue regardless or the input queue backs up.
void test_nonfinite_dt() {
    auto mp = in::Manager::create(); auto& M = *mp;
    const auto& W = M.key(KeyCode::W);

    // Build up some held_time, then feed a NaN frame.
    M.push_key(KeyCode::W, true); M.begin_frame(0.10f);
    CHECK(bs(W, true, true, false));  CHECK(ap(W.held_time, 0.0f));
    M.begin_frame(0.10f);             CHECK(ap(W.held_time, 0.10f));
    M.begin_frame(0.10f);             CHECK(ap(W.held_time, 0.20f));

    // NaN frame — held_time MUST NOT change (skip the bad accumulator).
    M.begin_frame(nan_f());
    CHECK(W.down);                    // input still down
    CHECK(ap(W.held_time, 0.20f));    // not poisoned to NaN

    // Subsequent finite frames resume accruing as normal.
    M.begin_frame(0.10f);             CHECK(ap(W.held_time, 0.30f));

    // Event drain MUST still have happened during the NaN frame — pump
    // a press through to verify queue isn't backed up.
    M.push_key(KeyCode::A, true);
    M.begin_frame(nan_f());           // NaN frame ingests A
    const auto& A = M.key(KeyCode::A);
    CHECK(bs(A, true, true, false));  // pressed pulse fires on the NaN frame
    CHECK(ap(A.held_time, 0.0f));     // fresh press → 0
}

// ---- stats: events_processed counts key/button drains only --------
void test_stats() {
    auto mp = in::Manager::create(); auto& M = *mp;
    auto s0 = M.stats();
    CHECK(s0.events_processed == static_cast<cardinal::u64>(0));
    CHECK(s0.keys_down_now == static_cast<cardinal::u64>(0));
    CHECK(s0.mouse_buttons_down == static_cast<cardinal::u64>(0));

    M.push_key(KeyCode::W, true);
    M.push_key(KeyCode::A, true);
    M.begin_frame(0.5f);
    auto s1 = M.stats();
    CHECK(s1.events_processed == static_cast<cardinal::u64>(2));
    CHECK(s1.keys_down_now == static_cast<cardinal::u64>(2));

    M.push_key(KeyCode::W, false); M.begin_frame(0.5f);
    auto s2 = M.stats();
    CHECK(s2.events_processed == static_cast<cardinal::u64>(3));
    CHECK(s2.keys_down_now == static_cast<cardinal::u64>(1));   // A only

    M.push_mouse_button(MouseButton::Right, true); M.begin_frame(0.5f);
    auto s3 = M.stats();
    CHECK(s3.events_processed == static_cast<cardinal::u64>(4));
    CHECK(s3.mouse_buttons_down == static_cast<cardinal::u64>(1));

    // Mouse move + wheel do NOT count as processed events.
    M.push_mouse_move(7, 7);
    M.push_mouse_wheel(2);
    M.begin_frame(0.5f);
    auto s4 = M.stats();
    CHECK(s4.events_processed == static_cast<cardinal::u64>(4));   // unchanged
    CHECK(s4.keys_down_now == static_cast<cardinal::u64>(1));
}

// ---- axis() must NEVER return NaN, even with NaN Binding::scale ----
// Binding::scale is a public float field with no setter (the user
// constructs `Binding{.kind, .code, .scale = ...}` directly). A
// NaN scale poisons the accumulator (NaN + x = NaN), and the two
// ordered clamps in axis() — `sum < -1` and `sum > 1` — are
// NaN-blind (NaN comparisons are unordered-false), so NaN survives
// both guards and the public API returns NaN. Documented contract
// (input.hpp:92-93): "clamped to [-1, +1]". NaN violates that —
// downstream consumers (UI ProgressBar, gameplay movement, anything
// computing position += axis*dt) propagate NaN. Same sanitize-at-
// boundary pattern as audio::play_3d volume/pitch and hud::bar fill.
void test_axis_nonfinite_scale() {
    auto mp = in::Manager::create(); auto& M = *mp;
    const float qnan = nan_f();

    // Single NaN-scale binding: holding the key should NOT propagate
    // NaN into the public API.
    M.bind_axis("Bad", in::Binding{ in::BindKind::Key,
        static_cast<cardinal::u32>(KeyCode::W), qnan });
    M.push_key(KeyCode::W, true);
    M.begin_frame(0.5f);
    const float v = M.axis("Bad");
    CHECK(v == v);                                      // finite (NaN != NaN)
    CHECK(v >= -1.0f && v <= 1.0f);                     // contract [-1, +1]

    // Mixed: one finite +1 + one NaN. Both held. Sum starts at +1,
    // then +=NaN poisons it. axis() must STILL return finite in range.
    M.bind_axis("Mix", kb(KeyCode::A, 1.0f));
    M.bind_axis("Mix", in::Binding{ in::BindKind::Key,
        static_cast<cardinal::u32>(KeyCode::S), qnan });
    M.push_key(KeyCode::A, true);
    M.push_key(KeyCode::S, true);
    M.begin_frame(0.5f);
    const float vm = M.axis("Mix");
    CHECK(vm == vm);                                    // finite
    CHECK(vm >= -1.0f && vm <= 1.0f);

    // Without the NaN-scale binding held, the finite +1 contribution
    // alone should still produce +1 cleanly.
    M.push_key(KeyCode::S, false);
    M.begin_frame(0.5f);
    CHECK(ap(M.axis("Mix"), 1.0f));
}

}  // namespace

int main() {
    test_keycode_strings();
    test_key_edges();
    test_key_taps();
    test_bounds();
    test_actions();
    test_axis();
    test_axis_nonfinite_scale();
    test_gameplay_gate();
    test_mouse();
    test_stats();
    test_nonfinite_dt();

    if (g_fail == 0) {
        cardinal::log::infof("inputtest", "OK  %d checks passed", g_checks);
        return 0;
    }
    cardinal::log::errorf("inputtest", "%d / %d checks FAILED",
                          g_fail, g_checks);
    return 1;
}
