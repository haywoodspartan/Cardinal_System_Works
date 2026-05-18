add_library(cardinal_compile STATIC
    src/compiler.cpp
    include/cardinal/compile/compiler.hpp)
add_library(cardinal::compile ALIAS cardinal_compile)

target_include_directories(cardinal_compile PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_compile
    PUBLIC  cardinal::core cardinal::rhi cardinal::compile_flags)

if(WIN32)
    # SEH-protected DXC wrapper requires /EHa, not the project-default /EHsc.
    # Confined to this single TU so the rest of the engine keeps the cheaper
    # synchronous-only EH model.
    target_compile_options(cardinal_compile PRIVATE /EHa)
    find_package(directx-dxc CONFIG REQUIRED)
    target_link_libraries(cardinal_compile PRIVATE Microsoft::DirectXShaderCompiler)
endif()

set_target_properties(cardinal_compile PROPERTIES FOLDER runtime)
