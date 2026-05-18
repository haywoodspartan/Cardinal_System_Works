add_executable(Cardinal_System_Window main.cpp)
target_link_libraries(Cardinal_System_Window PRIVATE
    cardinal::window cardinal::rhi cardinal::compile_flags)
set_target_properties(Cardinal_System_Window PROPERTIES FOLDER samples)

# Deploy any registered NVIDIA SDK runtime DLLs (DLSS, Reflex, NTC, etc.)
# next to the binary post-build. No-op if no SDKs detected.
cardinal_tp_deploy_runtimes(Cardinal_System_Window)
