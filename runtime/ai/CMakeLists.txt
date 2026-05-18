add_library(cardinal_ai STATIC
    src/ai.cpp
    include/cardinal/ai/ai.hpp)
add_library(cardinal::ai ALIAS cardinal_ai)

target_include_directories(cardinal_ai PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_ai
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_ai PROPERTIES FOLDER runtime)
