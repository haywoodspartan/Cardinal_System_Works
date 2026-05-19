set(CARDINAL_UI_HEADERS
    include/cardinal/ui/studio.hpp
    include/cardinal/ui/studio_engine.hpp
    # ImGui master reference template — the single sanctioned <imgui.h>
    # include site for the Studio function surface (panels + facade).
    # Mirrors the cardinal::core FOUNDATION RULE for the editor's UI dep.
    include/cardinal/ui/imgui.hpp
    # Free-function diagnostic panel — surfaces simd::active_backend()
    # + per-op binding + a micro-bench. No Studio class dependency, so
    # any host (StudioMin, custom apps) can call it directly.
    include/cardinal/ui/simd_panel.hpp
    # Free-function diagnostic panel — per-mesh vgeom drawn / master
    # tri ratio + attach button. Same shape as simd_panel.
    include/cardinal/ui/vgeom_panel.hpp
)

set(CARDINAL_UI_SOURCES
    src/studio.cpp
    # Studio decomposition step 1: the zero-StudioImpl-coupling Vulkan
    # diagnostic helpers (vk_result_name + throttled check_vk) carved
    # out of the studio.cpp god-file. Behaviour-identical; further
    # cohesive groups (gizmo math, panel forwarders) follow incrementally.
    src/studio_vk_diag.cpp
    # StudioEngine — the "roof" layer. Wraps cardinal::engine with the
    # Studio frame pump (dockspace, menu bar, panels, FlyCamera input).
    # Apps that want the editor experience link cardinal::ui and use
    # cardinal::ui::StudioEngine instead of cardinal::engine::Engine.
    src/studio_engine.cpp
    # Per-panel implementation files. Each panel lives in src/panels/<name>.cpp
    # with a small header in src/panels/<name>.hpp. Studio's class methods
    # forward to these via `cardinal::ui::panels::draw(...)`. Splitting the
    # panels out keeps each editable file under a few hundred lines so a
    # focused edit only re-compiles one TU.
    src/panels/actor_outliner.cpp
    src/panels/brush.cpp
    src/panels/class_picker.cpp
    src/panels/console.cpp
    src/panels/cppscript.cpp
    src/panels/curve_editor.cpp
    src/panels/editor_modes.cpp
    src/panels/game_bar.cpp
    src/panels/hierarchy.cpp
    src/panels/inspector.cpp
    src/panels/input_panel.cpp
    src/panels/log.cpp
    src/panels/memory.cpp
    src/panels/mesh_tools.cpp
    src/panels/mixer.cpp
    src/panels/nav_panel.cpp
    src/panels/particles_panel.cpp
    src/panels/profiler.cpp
    src/panels/save_load.cpp
    src/panels/sequencer.cpp
    src/panels/sim_bar.cpp
    src/panels/simd.cpp
    src/panels/vgeom.cpp
    src/panels/sky_panel.cpp
    src/panels/stats.cpp
    src/panels/texture_tools.cpp
    src/panels/vt.cpp
    src/panels/project_panel.cpp
    src/panels/cook_pack.cpp
    src/panels/shader_compiler.cpp
    src/panels/world_systems.cpp
    # We compile our own copy of imgui_impl_vulkan.cpp so it links against
    # volk's function pointers instead of the vcpkg `imgui[vulkan-binding]`
    # binary path — the latter crashed inside ImGui_ImplVulkan_Init under
    # Vulkan 1.3 because the prebuilt impl resolves vk* through vulkan-1.lib
    # while our process resolves them through volk's loader. Folding this
    # source in keeps function-pointer ownership in one place (volk) and
    # removes the prebuilt-binary as a moving part.
    external/imgui_backends/imgui_impl_vulkan.cpp
)
if(WIN32)
    list(APPEND CARDINAL_UI_SOURCES external/imgui_backends/imgui_impl_dx12.cpp)
endif()

add_library(cardinal_ui STATIC ${CARDINAL_UI_SOURCES} ${CARDINAL_UI_HEADERS})
add_library(cardinal::ui ALIAS cardinal_ui)

target_include_directories(cardinal_ui PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Our local copy of the imgui Vulkan backend lives in external/. We have to
# expose the directory privately so studio.cpp can include the local header
# (it picks up our overrides) instead of the vcpkg-installed one.
target_include_directories(cardinal_ui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui_backends
)

find_package(imgui          CONFIG REQUIRED)
find_package(VulkanHeaders  CONFIG REQUIRED)
find_package(volk           CONFIG REQUIRED)

target_link_libraries(cardinal_ui
    # cardinal::engine pulls in the entire wood layer transitively (window,
    # rhi, scene, render, trace, script, console, …, world). That keeps
    # this list short AND enforces the layering rule: anything Studio
    # reaches for must already be visible from the engine layer. If you
    # find yourself adding a direct link here, the dependency probably
    # belongs in cardinal::engine first.
    PUBLIC  cardinal::engine
            cardinal::compile_flags
    PRIVATE imgui::imgui Vulkan::Headers volk::volk
)
if(WIN32)
    target_link_libraries(cardinal_ui PRIVATE d3d12 dxgi)
endif()

# Tell our copy of imgui_impl_vulkan.cpp to skip Vulkan prototypes and route
# all calls through volk. studio.cpp already follows the same pattern.
target_compile_definitions(cardinal_ui PRIVATE
    VK_NO_PROTOTYPES
    IMGUI_IMPL_VULKAN_NO_PROTOTYPES
    IMGUI_IMPL_VULKAN_USE_VOLK
)

# Vendored ImGui backends are upstream sources — we don't touch them, so
# downgrade the engine's /WX policy for those translation units. Both
# files contain `HRESULT hr = ...; (void)hr;` patterns plus other warnings
# we don't want to chase upstream every refresh.
if(MSVC)
    set(_cardinal_imgui_backend_srcs
        external/imgui_backends/imgui_impl_vulkan.cpp
    )
    if(WIN32)
        list(APPEND _cardinal_imgui_backend_srcs
             external/imgui_backends/imgui_impl_dx12.cpp)
    endif()
    set_source_files_properties(${_cardinal_imgui_backend_srcs}
        PROPERTIES COMPILE_OPTIONS
        "/WX-;/wd4189;/wd4505;/wd4100;/wd4244;/wd4267")
endif()

set_target_properties(cardinal_ui PROPERTIES FOLDER runtime)
