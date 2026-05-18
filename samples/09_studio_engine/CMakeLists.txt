# Cardinal_System_StudioMin — minimal editor host built on the
# cardinal::ui::StudioEngine "roof" layer. Validates the wood/roof
# split end-to-end with ~150 lines of app code vs the legacy sample's
# ~2700.

add_executable(Cardinal_System_StudioMin main.cpp)

# StudioEngine pulls in the entire wood + roof via cardinal::ui
# (which transitively brings cardinal::engine + every runtime subsystem).
# We just need imgui::imgui for the in-app panel + rhi for the
# active_backend enum.
find_package(imgui CONFIG REQUIRED)

target_link_libraries(Cardinal_System_StudioMin PRIVATE
    cardinal::ui
    cardinal::engine
    cardinal::rhi
    # serial + sky/actor/game/sim are pulled in for the panel's Save/Load
    # buttons — MinApp owns its own sky::Sky + game::Game(+SimWorld+World)
    # and round-trips them via serial.
    cardinal::serial
    cardinal::sky
    cardinal::actor
    cardinal::game
    cardinal::sim
    cardinal::physics
    cardinal::net
    cardinal::compile_flags
    imgui::imgui)

set_target_properties(Cardinal_System_StudioMin PROPERTIES FOLDER samples)

# Deploy NVIDIA SDK runtime DLLs (DLSS, Reflex, etc.) next to the binary
# post-build — same pattern as the other sample executables.
cardinal_tp_deploy_runtimes(Cardinal_System_StudioMin)

# The plugin (Cardinal_System_HelloPlugin) depends on the studio sample.
# Make this minimal one depend on it too so a single-target build brings
# the plugin DLL along (it scans <exe>/plugins/).
if(TARGET Cardinal_System_HelloPlugin)
    add_dependencies(Cardinal_System_StudioMin Cardinal_System_HelloPlugin)
endif()
