add_library(cardinal_vt STATIC
    src/types.cpp
    src/tile_cache.cpp
    src/page_table.cpp
    src/decoder.cpp
    src/vt.cpp
    include/cardinal/vt/types.hpp
    include/cardinal/vt/tile_cache.hpp
    include/cardinal/vt/page_table.hpp
    include/cardinal/vt/decoder.hpp
    include/cardinal/vt/vt.hpp)
add_library(cardinal::vt ALIAS cardinal_vt)

target_include_directories(cardinal_vt PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_vt
    PUBLIC  cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_vt PROPERTIES FOLDER runtime)
