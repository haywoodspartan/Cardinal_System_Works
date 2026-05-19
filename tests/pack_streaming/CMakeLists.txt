# Async asset-STREAMING integration suite (pack↔async↔jobs). The pack
# suite defers load_async by design (needs the JobSystem + thread
# nondeterminism); the async suite pins the Future then()-vs-resolve
# contract load_async depends on but never drives the real path. This
# exercises pk::Archive::load_async end to end through a started
# JobSystem: N concurrent loads each resolve to exactly their own .cpk
# slice regardless of completion order (collected out-of-order), the
# in-flight counter is balanced, bytes_streamed reconciles exactly,
# the no-pool path resolves inline, and missing keys resolve empty
# (never hang). Schedule-independent invariants only. Also stresses the
# work-stealing deque backpressure (ac77f18) under realistic streaming
# fan-out. cardinal::pack brings cook+core transitively. Deterministic,
# headless (skips with no temp dir). Exit 0 = pass.
add_executable(Cardinal_Test_PackStreaming main.cpp)
target_link_libraries(Cardinal_Test_PackStreaming PRIVATE
    cardinal::pack
    cardinal::cook
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_PackStreaming PROPERTIES FOLDER tests)

add_test(NAME pack_streaming COMMAND Cardinal_Test_PackStreaming)
