add_library(cardinal_import STATIC
    src/import.cpp
    src/import_gltf.cpp
    src/import_megascans.cpp
    src/import_heightmap.cpp
    src/import_fbx.cpp
    src/json.hpp
    include/cardinal/import/import.hpp
    include/cardinal/import/to_asset.hpp)
add_library(cardinal::import ALIAS cardinal_import)

target_include_directories(cardinal_import PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# core+scene ONLY (deliberately NOT asset): asset→cook, so linking
# asset here would form an import→asset→cook cycle that blocks cook
# from delegating to this importer. The engine-asset-struct bridge is
# header-only (to_asset.hpp) — consumers that want it already link
# cardinal::asset and include that header. Pure CPU, zero third-party
# importer deps (OBJ/MTL hand-rolled, glTF zero-dep).
target_link_libraries(cardinal_import
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_import PROPERTIES FOLDER runtime)
