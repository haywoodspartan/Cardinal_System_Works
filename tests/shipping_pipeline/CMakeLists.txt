# Full-shipping-pipeline capstone. content_pipeline pinned
# source->cook->Builder->Registry for one blob; cook_incremental pinned
# the cook_all driver up to cooked/; dist pinned distribute() from a
# synthesised tree via Archive only. This spans the entire REAL chain:
# a real .obj through cook::cook_all (real importer+MeshCooker) ->
# pack::distribute (the real cooked/ tree) -> Archive::open ->
# asset::Registry::load_mesh, proving geometry is bit-exact end to end,
# the cook type header round-trips into the pack entry type, load_raw
# unwraps to the codec payload, and DEV (cooked/ dir mount) == SHIPPING
# (.cpk mount) for the same key. A seam regression there ships a broken
# client while every isolated suite stays green. Pure CPU,
# deterministic, headless (skips with no temp dir). Exit 0 = pass.
add_executable(Cardinal_Test_ShippingPipeline main.cpp)
target_link_libraries(Cardinal_Test_ShippingPipeline PRIVATE
    cardinal::asset
    cardinal::cook
    cardinal::pack
    cardinal::project
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_ShippingPipeline PROPERTIES FOLDER tests)

add_test(NAME shipping_pipeline COMMAND Cardinal_Test_ShippingPipeline)
