# End-to-end client-distribution integration suite — the capstone of
# the asset-pipeline arc. Composes the already-locked pieces
# (CookedAsset::serialize to craft a cooked tree, Archive to read the
# result) to prove pack::distribute() turns a cooked project into a
# shippable bundle: every cooked asset keyed + typed correctly,
# non-.cooked files ignored, invalid blobs typed Unknown but kept,
# saves bundled (nested paths, generic slashes), extra runtime files
# staged, and the distribution.cardinal manifest the launcher validates.
# Deterministic + headless (temp dirs). cardinal::pack + cardinal::cook
# bring their deps transitively; /W4 /WX via compile_flags.
add_executable(Cardinal_Test_Dist main.cpp)
target_link_libraries(Cardinal_Test_Dist PRIVATE
    cardinal::pack
    cardinal::cook
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Dist PROPERTIES FOLDER tests)

add_test(NAME dist COMMAND Cardinal_Test_Dist)
