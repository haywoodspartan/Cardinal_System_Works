add_library(cardinal_console STATIC
    src/console.cpp
    include/cardinal/console/console.hpp)
add_library(cardinal::console ALIAS cardinal_console)

target_include_directories(cardinal_console PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_console
    PUBLIC  cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_console PROPERTIES FOLDER runtime)
