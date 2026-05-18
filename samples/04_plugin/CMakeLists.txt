# Sample plugin DLL — drop into <bin>/plugins/ to be picked up by the host.
add_library(Cardinal_System_HelloPlugin SHARED hello_plugin.cpp)
add_library(cardinal::sample::hello_plugin ALIAS Cardinal_System_HelloPlugin)

target_link_libraries(Cardinal_System_HelloPlugin PRIVATE
    cardinal::plugin cardinal::core cardinal::compile_flags)

set_target_properties(Cardinal_System_HelloPlugin PROPERTIES
    FOLDER samples
    OUTPUT_NAME "Cardinal_System_HelloPlugin"
    PREFIX ""                     # avoid "lib" prefix on POSIX
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
)

# Per-config output dirs — Visual Studio multi-config generators otherwise
# drop the DLL into bin/plugins/Debug or bin/plugins/Release and the host
# (which scans <bin>/plugins/) won't find it. Force the same flat layout
# Ninja already uses so plugin behaviour is identical across generators.
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(Cardinal_System_HelloPlugin PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/bin/plugins
        RUNTIME_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/bin/plugins
    )
endforeach()

# Make the host depend on the plugin so a single-target build of
# `Cardinal_System_Studio` (debug or release) always brings the plugin
# along. The plugin doesn't link into the host — it's loaded at runtime —
# but CMake "depends_on" still triggers the build correctly.
if(TARGET Cardinal_System_Studio)
    add_dependencies(Cardinal_System_Studio Cardinal_System_HelloPlugin)
endif()
