# Deterministic AUTHORED-CONTENT pipeline INTEGRATION suite. Pins the
# link BEFORE the codec that asset_pipeline starts at: a real on-disk
# source mesh (tri.obj) through the importer + MeshCooker and out the
# far end of the shipping chain — cook → serialize → pack/.cpk →
# Archive::open → Registry::mount_archive + load_mesh. Proves imported
# geometry survives import+cook+pack+load bit-exactly, the cook is
# deterministic (the incremental-cook hash contract), and the Registry
# path equals a direct codec decode of the cooked payload. Pure CPU,
# deterministic, headless (skips with no temp dir). Exit 0 = pass.
add_executable(Cardinal_Test_ContentPipeline main.cpp)
target_link_libraries(Cardinal_Test_ContentPipeline PRIVATE
    cardinal::asset
    cardinal::cook
    cardinal::pack
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_ContentPipeline PROPERTIES FOLDER tests)

add_test(NAME content_pipeline COMMAND Cardinal_Test_ContentPipeline)
