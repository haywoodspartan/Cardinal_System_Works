# Deterministic cinematic-sequencer regression suite. cine::Player::tick
# is the playback backbone: it advances the play head and EDGE-TRIGGERS
# Audio/Event/Camera track events exactly once as the head crosses each
# event time (window: prev < t_event <= now). It also handles loop-wrap
# (split tail-pass + carry continuation, two distinct "previous" time
# trackers), non-looping clamp-and-stop, seek-clamp, and speed scaling.
# A regression silently breaks every cinematic — events double-fire,
# never fire, fire on the wrong lap, or the head runs off the end. Pure
# CPU, headless, fully deterministic. Event + Camera tracks are observed
# via recording callbacks; Audio (needs an audio::Engine) and Animation
# (needs an anim::Player) tracks share the identical window predicate
# and are intentionally out of scope here. Exit 0 = all pass.
add_executable(Cardinal_Test_Cine main.cpp)
target_link_libraries(Cardinal_Test_Cine PRIVATE
    cardinal::cine
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Cine PROPERTIES FOLDER tests)

add_test(NAME cine COMMAND Cardinal_Test_Cine)
