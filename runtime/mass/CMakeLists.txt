add_library(cardinal_mass STATIC
    src/mass.cpp
    include/cardinal/mass/mass.hpp)
add_library(cardinal::mass ALIAS cardinal_mass)

target_include_directories(cardinal_mass PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_mass
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_mass PROPERTIES FOLDER runtime)
