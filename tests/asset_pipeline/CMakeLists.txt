# Deterministic asset-pipeline INTEGRATION suite. Exercises the full
# shipping seam end to end — codec encode → cook::CookedAsset wrap →
# pack::Builder/.cpk → pack::Archive::open → asset::Registry decode —
# which no per-module suite covers. A seam regression silently ships
# unreadable assets even when every unit suite is green. Pure CPU,
# deterministic, headless (skips with no temp dir). Exit 0 = pass.
add_executable(Cardinal_Test_AssetPipeline main.cpp)
target_link_libraries(Cardinal_Test_AssetPipeline PRIVATE
    cardinal::asset
    cardinal::cook
    cardinal::pack
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_AssetPipeline PROPERTIES FOLDER tests)

add_test(NAME asset_pipeline COMMAND Cardinal_Test_AssetPipeline)
