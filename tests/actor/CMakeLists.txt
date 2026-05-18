# Deterministic actor-system regression suite. Actor + Component + World
# are the entity backbone the SimWorld iterates every tick. Pinned: the
# component lifecycle (on_attach at add, on_tick per Actor/World tick,
# on_detach at destruction) via external counters; monotonic actor ids;
# get_component type matching; the deferred destroy→sweep contract (kill
# marks dead, find/count still see it, tick skips it, sweep physically
# removes + fires on_detach); blueprints (factory + build, sorted names);
# the synchronous event bus (ordered fan-out, payload, unsubscribe);
# TagComponent add-dedupe/has/remove; built-in component defaults. A
# regression silently corrupts every gameplay object. Pure CPU, headless,
# fully deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_Actor main.cpp)
target_link_libraries(Cardinal_Test_Actor PRIVATE
    cardinal::actor
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Actor PROPERTIES FOLDER tests)

add_test(NAME actor COMMAND Cardinal_Test_Actor)
