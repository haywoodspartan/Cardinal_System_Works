# cardinal::ice — the modernized Softimage ICE: per-point attribute streams
# evaluated by a node graph that COMPILES to a flat topo-ordered op program
# with per-op dirty bits + output caches (the fast operator dependency graph:
# edits re-evaluate only their downstream cone). Pure CPU, deterministic.
add_library(cardinal_ice STATIC
    src/ice.cpp
    include/cardinal/ice/ice.hpp)
add_library(cardinal::ice ALIAS cardinal_ice)

target_include_directories(cardinal_ice PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_ice
    PUBLIC cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_ice PROPERTIES FOLDER runtime)
