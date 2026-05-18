# Deterministic engine-console regression suite. console::Registry is
# the one dispatch point for every runtime-tweakable knob (render flags,
# streaming radii, time scale, FPS caps) used by the Studio console,
# debug HUD and scripting shells. A regression silently breaks every
# cvar/command: wrong value parsed, missing clamp (range violation
# reaches live engine state), case-sensitivity drift, mis-tokenised
# quoted args, or a swallowed command. This suite pins the pure parser +
# registry: registration dedup / cross-kind collision, whole-string
# case-insensitive lookup, case-SENSITIVE sorted listings, prefix
# autocomplete (+ cap), the tokenizer's quoted-string rule, cvar
# read/print/set with int+float range clamping, command argv dispatch,
# the help/list built-ins, and Output truncation. Singleton is cleared
# per test; pure, headless, fully deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_Console main.cpp)
target_link_libraries(Cardinal_Test_Console PRIVATE
    cardinal::console
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Console PROPERTIES FOLDER tests)

add_test(NAME console COMMAND Cardinal_Test_Console)
