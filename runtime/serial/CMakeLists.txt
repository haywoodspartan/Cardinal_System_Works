add_library(cardinal_serial STATIC
    src/serial.cpp
    include/cardinal/serial/serial.hpp)
add_library(cardinal::serial ALIAS cardinal_serial)

target_include_directories(cardinal_serial PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_serial
    PUBLIC  cardinal::core cardinal::actor cardinal::game cardinal::sky
            cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_serial PROPERTIES FOLDER runtime)
