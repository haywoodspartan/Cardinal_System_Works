add_executable(Cardinal_Sample_SlateNative main.cpp)
target_link_libraries(Cardinal_Sample_SlateNative PRIVATE
    cardinal::window
    cardinal::input
    cardinal::cui_rhi          # brings cardinal::cui + cardinal::rhi transitively
    cardinal::compile_flags)
set_target_properties(Cardinal_Sample_SlateNative PROPERTIES FOLDER samples)

# Deploy any registered NVIDIA SDK runtime DLLs next to the binary (no-op if none).
cardinal_tp_deploy_runtimes(Cardinal_Sample_SlateNative)
