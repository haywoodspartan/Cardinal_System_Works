add_library(cardinal_asset STATIC
    src/asset.cpp
    include/cardinal/asset/asset.hpp)
add_library(cardinal::asset ALIAS cardinal_asset)

target_include_directories(cardinal_asset PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_asset
    PUBLIC  cardinal::core cardinal::scene cardinal::cook cardinal::pack
            cardinal::compile_flags)

set_target_properties(cardinal_asset PROPERTIES FOLDER runtime)
