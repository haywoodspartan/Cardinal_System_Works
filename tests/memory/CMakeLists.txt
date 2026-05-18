# Deterministic memory-pressure regression suite. classify() is the
# budget-decision keystone: every subsystem's grow/hold/trim/evict
# reaction (and the StudioMin memory panel) hinges on the used_pct →
# Pressure mapping. A boundary regression silently mis-budgets — never
# reaching Critical (OOM) or always Critical (refuses allocs). That
# math is pure + exhaustively lockable; the OS-snapshot / wall-clock
# Monitor parts get invariant + self-consistency + interval-gate
# checks (absolute RAM varies, so no value assertions there).
add_executable(Cardinal_Test_Memory main.cpp)
target_link_libraries(Cardinal_Test_Memory PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Memory PROPERTIES FOLDER tests)

add_test(NAME memory COMMAND Cardinal_Test_Memory)
