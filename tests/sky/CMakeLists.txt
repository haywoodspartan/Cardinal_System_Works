# Deterministic sky time-of-day regression suite. Sky::set_hour →
# recompute_state_ blends the phase keys into the scene's lighting
# (sun_dir / sun_color / ambient / zenith / horizon). serial_world only
# round-trips the KEYS; this pins the interpolation that actually
# produces light — a regression mis-lights the entire scene
# (wrong sun angle / colour / day-night blend), invisible until it
# ships. Pure CPU, headless, fully deterministic; cardinal hours
# (0/6/12/18) have exact trig so sun_dir is pinned without <cmath>.
add_executable(Cardinal_Test_Sky main.cpp)
target_link_libraries(Cardinal_Test_Sky PRIVATE
    cardinal::sky
    cardinal::scene
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Sky PROPERTIES FOLDER tests)

add_test(NAME sky COMMAND Cardinal_Test_Sky)
