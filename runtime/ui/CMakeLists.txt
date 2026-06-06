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
    # Free-function Options / Settings panel — categorised editor over the
    # console CVar registry + Save/Load to a settings file. Same shape.
    include/cardinal/ui/options_panel.hpp
)

set(CARDINAL_UI_SOURCES
    src/studio.cpp
    # Studio decomposition step 1: the zero-StudioImpl-coupling Vulkan
    # diagnostic helpers (vk_result_name + throttled check_vk) carved
    # out of the studio.cpp god-file. Behaviour-identical; further
    # cohesive groups (gizmo math, panel forwarders) follow incrementally.
    src/studio_vk_diag.cpp
    # Studio decomposition step 2: the pure gizmo-math primitives
    # (project_/mouse_world_ray_/ray_plane_) — verbatim, member-free,
    # behaviour-identical; StudioImpl keeps thin forwarders.
    src/studio_gizmo_math.cpp
    # StudioEngine — the "roof" layer. Wraps cardinal::engine with the
    # Studio frame pump (dockspace, menu bar, panels, FlyCamera input).
    # Apps that want the editor experience link cardinal::ui and use
    # cardinal::ui::StudioEngine instead of cardinal::engine::Engine.
    src/studio_engine.cpp
    # Per-panel implementation files, grouped by Studio concern into
    # src/panels/<area>/<name>.cpp (+ a small same-dir <name>.hpp). Studio's
    # class methods forward to these via `cardinal::ui::panels::draw(...)`.
    # Splitting the panels out — and now foldering them by area — keeps each
    # editable file small and the tree navigable by feature.
    #  scene/      — actor/scene authoring
    src/panels/scene/actor_outliner.cpp
    src/panels/scene/hierarchy.cpp
    src/panels/scene/inspector.cpp
    src/panels/scene/prefab.cpp
    src/panels/scene/class_picker.cpp
    #  assets/     — project / cook+pack / save-load
    src/panels/assets/project_panel.cpp
    src/panels/assets/cook_pack.cpp
    src/panels/assets/save_load.cpp
    #  gameplay/   — play / sim / world systems / input / nav
    src/panels/gameplay/game_bar.cpp
    src/panels/gameplay/sim_bar.cpp
    src/panels/gameplay/world_systems.cpp
    src/panels/gameplay/input_panel.cpp
    src/panels/gameplay/nav_panel.cpp
    #  animation/  — sequencer / curves / audio mixer
    src/panels/animation/sequencer.cpp
    src/panels/animation/curve_editor.cpp
    src/panels/animation/mixer.cpp
    #  tools/      — brush / mesh + texture tools / editor modes
    src/panels/tools/brush.cpp
    src/panels/tools/mesh_tools.cpp
    src/panels/tools/texture_tools.cpp
    src/panels/tools/editor_modes.cpp
    #  rendering/  — sky / virtual textures / vgeom / particles
    src/panels/rendering/sky_panel.cpp
    src/panels/rendering/vt.cpp
    src/panels/rendering/vgeom.cpp
    src/panels/rendering/particles_panel.cpp
    #  scripting/  — C++ sandbox / shader compiler
    src/panels/scripting/cppscript.cpp
    src/panels/scripting/shader_compiler.cpp
    #  diagnostics/ — log / console / stats / memory / profiler / simd
    src/panels/diagnostics/log.cpp
    src/panels/diagnostics/console.cpp
    src/panels/diagnostics/stats.cpp
    src/panels/diagnostics/memory.cpp
    src/panels/diagnostics/profiler.cpp
    src/panels/diagnostics/simd.cpp
    src/panels/diagnostics/options.cpp
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
# ImGuizmo — 3D translate/rotate/scale handles + ViewManipulate orbit cube
# layered on top of ImGui. Wired into draw_viewport_panel; coexists with
# the existing hand-rolled gizmo (toggled via studio_impl flag).
find_package(imguizmo       CONFIG REQUIRED)

target_link_libraries(cardinal_ui
    # cardinal::engine pulls in the entire wood layer transitively (window,
    # rhi, scene, render, trace, script, console, …, world). That keeps
    # this list short AND enforces the layering rule: anything Studio
    # reaches for must already be visible from the engine layer. If you
    # find yourself adding a direct link here, the dependency probably
    # belongs in cardinal::engine first.
    PUBLIC  cardinal::engine
            cardinal::compile_flags
    PRIVATE imgui::imgui imguizmo::imguizmo Vulkan::Headers volk::volk
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
