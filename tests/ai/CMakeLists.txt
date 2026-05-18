# Deterministic AI-core regression suite. Three pure, headless pieces:
#
#   Blackboard    — typed std::variant store. The headline trap is the
#                   TYPE-MISMATCH read: get_float on an int key must
#                   return the default, not a coerced value.
#   Behavior tree — Sequence/Selector are STATEFUL: a Running child
#                   suspends the cursor and earlier children must NOT
#                   re-run next tick. A regression there silently
#                   re-fires actions every frame (re-spawn, re-path).
#   Perception    — sight cone (cos-angle + radius + co-located degen),
#                   hearing (radius × sound strength), stimulus lifetime
#                   aging, deferred remove-compaction, event ordering.
#                   A regression makes every agent blind or omniscient.
#
# All math is exact (Pythagorean / axis-aligned geometry) so no <cmath>
# in the test and no FP-boundary fragility. Pure CPU, fully
# deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_AI main.cpp)
target_link_libraries(Cardinal_Test_AI PRIVATE
    cardinal::ai
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_AI PROPERTIES FOLDER tests)

add_test(NAME ai COMMAND Cardinal_Test_AI)
