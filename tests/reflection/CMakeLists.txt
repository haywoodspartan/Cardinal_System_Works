# Deterministic game-reflection regression suite. ClassRegistry +
# CARDINAL_REGISTER_GAME_CLASS are the editor/serialization backbone:
# every GameActor subclass advertises a factory + a property descriptor
# list (name/kind/min/max/tooltip + a void* into the live instance) so
# the inspector can read/write fields and save/load can round-trip them.
# A regression silently breaks property editing and on-disk class data:
# wrong ptr (writes corrupt the wrong field), lost class (spawn fails),
# mis-sorted picker, or a broken last-wins reload. Pure CPU, fully
# deterministic; the registry is a process singleton so tests are
# baseline-relative + use unique class names. Exit 0 = all pass.
add_executable(Cardinal_Test_Reflection main.cpp)
target_link_libraries(Cardinal_Test_Reflection PRIVATE
    cardinal::game
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Reflection PROPERTIES FOLDER tests)

add_test(NAME reflection COMMAND Cardinal_Test_Reflection)
