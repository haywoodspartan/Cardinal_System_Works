# Deterministic cooker regression suite. cook is the middle of the
# asset pipeline (import → COOK → pack): the CookedAsset typed-blob
# format is exactly what pack/distribute reads the type header from, so
# a codec regression here silently mis-types or corrupts every shipped
# asset. Covers the serialize/deserialize round-trip + exact on-disk
# header + failure paths, the asset-type lookups, and the MeshCooker
# import→cook delegation path (+ CARDMESH + box fallback) including
# build-determinism. cook_all/Driver (needs a project::Project tree)
# and the texture/shader/audio cookers (image decode / shader::Compiler)
# are out of scope here by design — they pair with the deferred
# pack distribute() integration arc. cardinal::cook brings its deps
# transitively; /W4 /WX via compile_flags.
add_executable(Cardinal_Test_Cook main.cpp)
target_link_libraries(Cardinal_Test_Cook PRIVATE
    cardinal::cook
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Cook PROPERTIES FOLDER tests)

add_test(NAME cook COMMAND Cardinal_Test_Cook)
