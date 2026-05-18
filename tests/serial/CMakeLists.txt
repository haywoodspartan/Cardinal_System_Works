# Deterministic serialization regression suite. A save-format
# regression silently corrupts user data — the highest-stakes "looks
# fine, ships, eats saves" surface. Covers the pure text:: format
# primitives (exact-string round-trips) and the full Sky save/load
# round-trip (self-contained CPU object — no rhi::Device, unlike
# Scene/Mesh which a headless test can't construct). World/Scene
# round-trips need an actor ClassRegistry / a GPU device respectively
# and are out of scope here by design. cardinal::serial brings its
# deps transitively; /W4 /WX via compile_flags.
add_executable(Cardinal_Test_Serial main.cpp)
target_link_libraries(Cardinal_Test_Serial PRIVATE
    cardinal::serial
    cardinal::sky
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Serial PROPERTIES FOLDER tests)

add_test(NAME serial COMMAND Cardinal_Test_Serial)
