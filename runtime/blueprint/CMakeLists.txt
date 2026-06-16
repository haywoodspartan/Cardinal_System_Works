# cardinal::blueprint — interoperable visual scripting. The round-trip engine
# between a node GRAPH and editable SOURCE TEXT (a small purpose-built DSL):
# generate_source (graph -> text) + parse_source (text -> graph, flattening
# nested expressions into discrete nodes). Pure CPU; execution (DSL -> VM
# bytecode) is a later slice, so no vm dep yet.
add_library(cardinal_blueprint STATIC
    src/blueprint.cpp
    src/compile_vm.cpp
    include/cardinal/blueprint/blueprint.hpp)
add_library(cardinal::blueprint ALIAS cardinal_blueprint)

target_include_directories(cardinal_blueprint PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# vm is PRIVATE: only the execution backend (compile_vm.cpp) uses it; the
# graph<->source API stays VM-free.
target_link_libraries(cardinal_blueprint
    PUBLIC  cardinal::core cardinal::compile_flags
    PRIVATE cardinal::vm)

set_target_properties(cardinal_blueprint PROPERTIES FOLDER runtime)
