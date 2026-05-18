# Deterministic input-manager regression suite. input::Manager is a pure
# state machine fed by push_*() (exactly how the OS hook feeds it) and
# advanced by begin_frame(dt). The contract that gameplay relies on:
#
#   * pressed/released are ONE-frame rising/falling-edge pulses;
#   * held_time accumulates (in the pre-pass) only while down, including
#     the release frame, and resets to 0 ONLY on a fresh press — never
#     on release (it goes stale, by design);
#   * an action is the OR of its bindings; an axis is the clamped sum of
#     scaled contributions;
#   * the PIE gameplay gate zeroes action_*/axis but NOT direct key/
#     mouse queries (Studio hotkeys must survive);
#   * mouse delta/wheel are per-frame; queries before begin_frame see
#     nothing (events are queued, drained on begin_frame).
#
# A regression here silently breaks every control. The Win32 hook
# (input_windows.cpp) is OS-bound and out of scope — the producer-side
# push_* API drives the identical state path. Pure, headless,
# deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_Input main.cpp)
target_link_libraries(Cardinal_Test_Input PRIVATE
    cardinal::input
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Input PROPERTIES FOLDER tests)

add_test(NAME input COMMAND Cardinal_Test_Input)
