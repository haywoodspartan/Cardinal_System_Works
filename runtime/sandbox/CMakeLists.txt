add_library(cardinal_sandbox STATIC
    src/sandbox.cpp
    include/cardinal/sandbox/sandbox.hpp)
add_library(cardinal::sandbox ALIAS cardinal_sandbox)

target_include_directories(cardinal_sandbox PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_sandbox
    PUBLIC  cardinal::core cardinal::plugin cardinal::compile_flags)

set_target_properties(cardinal_sandbox PROPERTIES FOLDER runtime)
