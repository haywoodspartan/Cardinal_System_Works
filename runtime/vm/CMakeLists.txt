add_library(cardinal_vm STATIC
    src/vm.cpp
    include/cardinal/vm/vm.hpp)
add_library(cardinal::vm ALIAS cardinal_vm)

target_include_directories(cardinal_vm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_vm
    PUBLIC  cardinal::core cardinal::compile_flags)

set_target_properties(cardinal_vm PROPERTIES FOLDER runtime)
