add_library(cardinal_partition STATIC
    src/partition.cpp
    include/cardinal/partition/partition.hpp)
add_library(cardinal::partition ALIAS cardinal_partition)

target_include_directories(cardinal_partition PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_partition
    PUBLIC  cardinal::core cardinal::scene
            cardinal::compile_flags)

set_target_properties(cardinal_partition PROPERTIES FOLDER runtime)
