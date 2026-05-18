add_library(cardinal_hud STATIC
    src/hud.cpp
    include/cardinal/hud/hud.hpp)
add_library(cardinal::hud ALIAS cardinal_hud)

target_include_directories(cardinal_hud PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(cardinal_hud
    PUBLIC  cardinal::core cardinal::compile_flags
    PRIVATE imgui::imgui)

set_target_properties(cardinal_hud PROPERTIES FOLDER runtime)
