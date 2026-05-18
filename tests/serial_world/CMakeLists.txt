# Deterministic world save/load round-trip — the highest-stakes save
# surface ("actors are the primary game state"). Defines + registers a
# reflected test GameActor exercising every PropertyKind, round-trips a
# World through save_world/load_world, and asserts identity + transform
# + every property survives exactly. Also the regression lock for the
# string-quote-accretion fix in serial.cpp. Headless + deterministic
# (no rhi::Device): SimWorld/Game/World are pure CPU. Explicit deps so
# every symbol resolves regardless of transitive-link ordering.
add_executable(Cardinal_Test_SerialWorld main.cpp)
target_link_libraries(Cardinal_Test_SerialWorld PRIVATE
    cardinal::serial
    cardinal::game
    cardinal::sim
    cardinal::actor
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_SerialWorld PROPERTIES FOLDER tests)

add_test(NAME serial_world COMMAND Cardinal_Test_SerialWorld)
