add_library(cardinal_plugin STATIC
    src/plugin.cpp
    include/cardinal/plugin/plugin.hpp)
add_library(cardinal::plugin ALIAS cardinal_plugin)

target_include_directories(cardinal_plugin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_plugin
    PUBLIC  cardinal::core cardinal::trace cardinal::render cardinal::compile_flags)

if(WIN32)
    # Plugin tick is SEH-wrapped — needs /EHa rather than /EHsc.
    target_compile_options(cardinal_plugin PRIVATE /EHa)
endif()

set_target_properties(cardinal_plugin PROPERTIES FOLDER runtime)
