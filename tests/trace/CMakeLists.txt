# Deterministic trace-timeline ring-buffer regression suite. Timeline is
# the lock-free SPSC ring behind the Function Tracer panel + replay
# record: a fixed kCap ring with a monotonic write index. snapshot()
# returns the most-recent min(written,kCap) events oldest-first via
# (written-count)%kCap modular indexing; the ring is lossy-by-design
# (oldest events fall off on overflow). A regression silently corrupts
# the profiler/replay record — wrong event order, lost-too-early, or a
# stale snapshot. The push/snapshot/clear/wrap/eviction math + name
# truncation are pure and (single-writer) fully deterministic. t_ns
# (wall clock) and thread_id (OS) are asserted only as monotonic /
# self-consistent, never by absolute value. DbgHelp stack.cpp is
# environment-bound and intentionally out of scope. Exit 0 = all pass.
add_executable(Cardinal_Test_Trace main.cpp)
target_link_libraries(Cardinal_Test_Trace PRIVATE
    cardinal::trace
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Trace PROPERTIES FOLDER tests)

add_test(NAME trace COMMAND Cardinal_Test_Trace)
