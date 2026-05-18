add_library(cardinal_pack STATIC
    src/pack.cpp
    include/cardinal/pack/pack.hpp)
add_library(cardinal::pack ALIAS cardinal_pack)

target_include_directories(cardinal_pack PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_pack
    PUBLIC  cardinal::core cardinal::cook cardinal::compile_flags)

set_target_properties(cardinal_pack PROPERTIES FOLDER runtime)
