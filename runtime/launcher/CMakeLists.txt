set(CARDINAL_LAUNCHER_HEADERS
    include/cardinal/launcher/startup.hpp
)

set(CARDINAL_LAUNCHER_SOURCES)
if(WIN32)
    list(APPEND CARDINAL_LAUNCHER_SOURCES src/startup_windows.cpp)
else()
    list(APPEND CARDINAL_LAUNCHER_SOURCES src/startup_linux.cpp)
endif()

add_library(cardinal_launcher STATIC
    ${CARDINAL_LAUNCHER_SOURCES}
    ${CARDINAL_LAUNCHER_HEADERS})
add_library(cardinal::launcher ALIAS cardinal_launcher)

target_include_directories(cardinal_launcher PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(cardinal_launcher
    PUBLIC  cardinal::core cardinal::rhi cardinal::compile_flags
)

if(WIN32)
    # Win32 launcher dialog probes Vulkan via vulkan-1.dll loader symbols and
    # D3D12 via d3d12.dll + dxgi.dll. We delay-load nothing — both DLLs are
    # checked with LoadLibraryA so missing-DLL is just "backend unavailable".
    # Visuals are owner-drawn with GDI+ (anti-aliased text + rounded rects).
    find_package(VulkanHeaders CONFIG REQUIRED)
    target_link_libraries(cardinal_launcher PUBLIC Vulkan::Headers)
    target_link_libraries(cardinal_launcher PRIVATE d3d12 dxgi gdiplus)
endif()

set_target_properties(cardinal_launcher PROPERTIES FOLDER runtime)
