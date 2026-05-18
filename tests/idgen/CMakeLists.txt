# Deterministic Handle / SlotPool / IdGen regression suite. This is the
# stale-handle / use-after-free guard under physics BodyHandle, every
# tag-typed handle, and the threadsafe id source. A generation-tracking
# regression silently resolves a dead handle to the WRONG live object
# (memory-safety-adjacent, gameplay-corrupting). Pure + header-only;
# the IdGen concurrency check uses std::thread but asserts only
# schedule-independent set properties so it stays deterministic.
add_executable(Cardinal_Test_IdGen main.cpp)
target_link_libraries(Cardinal_Test_IdGen PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_IdGen PROPERTIES FOLDER tests)

add_test(NAME idgen COMMAND Cardinal_Test_IdGen)
