add_library(cardinal_script STATIC
    src/engine.cpp
    include/cardinal/script/engine.hpp)
add_library(cardinal::script ALIAS cardinal_script)

target_include_directories(cardinal_script PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# Lua 5.4 via vcpkg manifest. find_package handles both `lua` and `Lua`.
find_package(Lua REQUIRED)

target_link_libraries(cardinal_script
    PUBLIC  cardinal::core cardinal::trace cardinal::compile_flags
    PRIVATE ${LUA_LIBRARIES})
target_include_directories(cardinal_script PRIVATE ${LUA_INCLUDE_DIR})

set_target_properties(cardinal_script PROPERTIES FOLDER runtime)
