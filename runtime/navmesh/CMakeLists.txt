add_library(cardinal_navmesh STATIC
    src/navmesh.cpp
    include/cardinal/navmesh/navmesh.hpp)
add_library(cardinal::navmesh ALIAS cardinal_navmesh)

target_include_directories(cardinal_navmesh PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_navmesh
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_navmesh PROPERTIES FOLDER runtime)
