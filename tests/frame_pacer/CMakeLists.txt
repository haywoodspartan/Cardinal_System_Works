# Deterministic frame-pacer regression suite. FramePacer arbitrates the
# engine's whole frame budget: every render context (main loop + each
# editor viewport) registers an FPS quota and the pacer applies the
# STRICTEST — minimum non-zero — cap across active contexts each frame
# (a 0/negative cap means "uncapped", contributing nothing). A
# regression silently mis-paces the entire engine: too-low cap → visible
# stutter, lost cap → 100% CPU/GPU spin + thermal/battery burn. The
# cap-selection + limit registry are pure and fully deterministic; the
# OS sleep duration and the Win32 foreground probe are wall-clock /
# environment dependent and intentionally OUT OF SCOPE (the foreground
# path is pinned only in its window-unset "always foreground" form, and
# only the uncapped no-sleep return is asserted). Pure CPU, headless.
add_executable(Cardinal_Test_FramePacer main.cpp)
target_link_libraries(Cardinal_Test_FramePacer PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_FramePacer PROPERTIES FOLDER tests)

add_test(NAME frame_pacer COMMAND Cardinal_Test_FramePacer)
