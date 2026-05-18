add_library(cardinal_trace STATIC
    src/stack.cpp
    src/timeline.cpp
    include/cardinal/trace/stack.hpp
    include/cardinal/trace/timeline.hpp)
add_library(cardinal::trace ALIAS cardinal_trace)

target_include_directories(cardinal_trace PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_trace PUBLIC cardinal::core cardinal::compile_flags)

if(WIN32)
    target_link_libraries(cardinal_trace PRIVATE DbgHelp)
endif()

set_target_properties(cardinal_trace PROPERTIES FOLDER runtime)
