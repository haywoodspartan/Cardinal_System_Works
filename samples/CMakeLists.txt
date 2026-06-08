add_subdirectory(00_topology)
add_subdirectory(01_jobs)
add_subdirectory(02_window)
add_subdirectory(03_studio)
add_subdirectory(04_plugin)
add_subdirectory(05_terrain_dump)
add_subdirectory(07_cppscript)
add_subdirectory(08_sandbox)
# Minimal editor host on the cardinal::ui::StudioEngine "roof" layer —
# validates the wood/roof split with a tiny app body.
add_subdirectory(09_studio_engine)
# Capability-sandboxed bytecode VM, end to end (verify -> child process ->
# tick -> host-call over the pipe).
add_subdirectory(10_vm_sandbox)
# Codebase stress world — heavy actor/sim/game + PlayerController +
# particles + virtual texturing + net loopback + navmesh every frame.
add_subdirectory(11_stress_world)
# Cardinal Slate native UI — our own retained-mode framework as a native,
# GPU-rendered windowed app (no ImGui): window + input + RHI renderer.
add_subdirectory(12_slate_native)
