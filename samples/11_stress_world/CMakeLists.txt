# Cardinal_System_StressWorld — a deliberately heavy StudioApplication that
# drives actor/sim/game + the reusable PlayerControllerComponent + particles
# + virtual texturing + net loopback replication + navmesh every frame so
# throughput / lifetime / integration regressions surface immediately.

add_executable(Cardinal_System_StressWorld main.cpp)

find_package(imgui CONFIG REQUIRED)

target_link_libraries(Cardinal_System_StressWorld PRIVATE
    cardinal::ui
    cardinal::engine
    cardinal::rhi
    cardinal::actor
    cardinal::game
    cardinal::sim
    cardinal::scene
    cardinal::particles
    cardinal::vt
    cardinal::net
    cardinal::navmesh
    cardinal::compile_flags
    imgui::imgui)

set_target_properties(Cardinal_System_StressWorld PROPERTIES FOLDER samples)

# Deploy NVIDIA SDK runtime DLLs next to the binary (same pattern as the
# other sample executables).
cardinal_tp_deploy_runtimes(Cardinal_System_StressWorld)
