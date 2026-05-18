add_library(cardinal_cppscript STATIC
    src/cppscript.cpp
    include/cardinal/cppscript/cppscript.hpp)
add_library(cardinal::cppscript ALIAS cardinal_cppscript)

target_include_directories(cardinal_cppscript PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_cppscript
    PUBLIC  cardinal::core cardinal::plugin cardinal::compile_flags)

set_target_properties(cardinal_cppscript PROPERTIES FOLDER runtime)
