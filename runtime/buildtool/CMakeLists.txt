# cardinal::buildtool — the BuildCookRun pipeline orchestrator. Sits above
# project / cook / pack (cook depends on project, so the orchestrator can't
# live inside project) and composes them into a one-call shippable-build
# pipeline. Cardinal's analog of Unreal's BuildCookRun / UAT.
add_library(cardinal_buildtool STATIC
    src/pipeline.cpp
    include/cardinal/buildtool/pipeline.hpp)
add_library(cardinal::buildtool ALIAS cardinal_buildtool)

target_include_directories(cardinal_buildtool PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_buildtool
    PUBLIC  cardinal::core cardinal::project cardinal::cook cardinal::pack
            cardinal::compile_flags)

set_target_properties(cardinal_buildtool PROPERTIES FOLDER runtime)
