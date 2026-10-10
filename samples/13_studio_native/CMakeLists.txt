# Cardinal Studio (native) — the editor shell on Cardinal Slate, with NO
# ImGui: window + input + RHI device/swapchain + scene::ForwardRenderer for
# the 3D view, and cardinal::cui_shell (menu bar, WindowManager desktop,
# Hierarchy / Inspector / Console / Log / Stats) drawn by the native cui_rhi
# renderer on top.
add_executable(Cardinal_System_StudioNative main.cpp)
target_link_libraries(Cardinal_System_StudioNative PRIVATE
    cardinal::window
    cardinal::input
    cardinal::scene
    cardinal::console
    cardinal::cui_shell
    cardinal::cui_rhi
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_System_StudioNative PROPERTIES FOLDER samples)

# Deploy any registered NVIDIA SDK runtime DLLs next to the binary (no-op if none).
cardinal_tp_deploy_runtimes(Cardinal_System_StudioNative)
