# Deterministic audio-engine regression suite. cardinal::audio::Engine is
# the device-agnostic core (channel tree, cue catalog, instance
# lifecycle, 3D attenuation, PCM render) — the WASAPI OutputBackend is
# the only OS-bound piece and is out of scope. A regression here silently
# breaks every game sound: wrong mix gain, uncontrollable instances,
# attenuation drift. Pure CPU, fully deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_Audio main.cpp)
target_link_libraries(Cardinal_Test_Audio PRIVATE
    cardinal::audio
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Audio PROPERTIES FOLDER tests)

add_test(NAME audio COMMAND Cardinal_Test_Audio)
