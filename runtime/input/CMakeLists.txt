set(CARDINAL_INPUT_SRC src/input.cpp)
if(WIN32)
    list(APPEND CARDINAL_INPUT_SRC src/input_windows.cpp)
endif()

add_library(cardinal_input STATIC
    ${CARDINAL_INPUT_SRC}
    include/cardinal/input/input.hpp)
add_library(cardinal::input ALIAS cardinal_input)

target_include_directories(cardinal_input PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_input
    PUBLIC  cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_input PROPERTIES FOLDER runtime)
