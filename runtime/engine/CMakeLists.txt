# =============================================================================
# cardinal::engine — the "wood" layer.
#
# Layering model:
#   cardinal::core            (concrete foundation)
#   cardinal::engine          (wood — built on the concrete)        ← this
#   cardinal::ui::StudioEngine (roof — built on the wood, lives in runtime/ui)
#
# This target aggregates every runtime subsystem that does NOT depend on
# the editor / Studio. A shipped game links cardinal::engine directly and
# never pays for ImGui or panel code; an editor-flavoured host links
# cardinal::ui::StudioEngine which transitively brings the wood in.
#
# Hard rule: this target must NOT link cardinal::ui or any imgui::*. If
# you find yourself needing UI from inside an engine subsystem, you've
# almost certainly written editor code that belongs in runtime/ui/.
# =============================================================================
add_library(cardinal_engine STATIC
    src/engine.cpp
    include/cardinal/engine/engine.hpp)
add_library(cardinal::engine ALIAS cardinal_engine)

target_include_directories(cardinal_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# Direct dependencies — what engine.cpp itself includes.
target_link_libraries(cardinal_engine
    PUBLIC  cardinal::core
            cardinal::window cardinal::rhi
            cardinal::scene  cardinal::render
            cardinal::script cardinal::plugin
            cardinal::physics
            cardinal::compile_flags

    # Wood-layer aggregation — anything that links cardinal::engine
    # transitively pulls in every non-UI runtime subsystem. This keeps the
    # call sites simple (apps say "give me the engine") and enforces the
    # layering: if a subsystem links cardinal::ui by mistake, this target's
    # dependency closure blows up at link time.
    PUBLIC  cardinal::actor    cardinal::sim
            cardinal::anim     cardinal::audio
            cardinal::cine     cardinal::game
            cardinal::sky      cardinal::input
            cardinal::serial   cardinal::particles
            cardinal::nav      cardinal::hud
            cardinal::project  cardinal::shader
            cardinal::cook     cardinal::pack
            cardinal::asset    cardinal::vt
            cardinal::edit     cardinal::console
            cardinal::cppscript cardinal::sandbox
            cardinal::partition cardinal::level
            cardinal::mass     cardinal::navmesh
            cardinal::ai       cardinal::trace
            cardinal::world    cardinal::vgeom)

set_target_properties(cardinal_engine PROPERTIES FOLDER runtime)
