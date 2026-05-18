add_library(cardinal_sim STATIC
    src/sim.cpp
    include/cardinal/sim/sim.hpp)
add_library(cardinal::sim ALIAS cardinal_sim)

target_include_directories(cardinal_sim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_sim
    PUBLIC  cardinal::core cardinal::actor cardinal::compile_flags)

set_target_properties(cardinal_sim PROPERTIES FOLDER runtime)
