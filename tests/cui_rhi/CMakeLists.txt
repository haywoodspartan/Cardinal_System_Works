# cui_rhi native renderer geometry suite — pins Renderer::build_geometry
# (DrawList -> vertex stream): counts, positions, packed color, font expansion.
# Pure/device-free, headless. Exit 0 = pass.
add_executable(Cardinal_Test_CuiRhi main.cpp)
target_link_libraries(Cardinal_Test_CuiRhi PRIVATE
    cardinal::cui_rhi
    cardinal::cui
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_CuiRhi PROPERTIES FOLDER tests)

add_test(NAME cui_rhi COMMAND Cardinal_Test_CuiRhi)
