# Deterministic incremental-cook DRIVER suite. The cook suite tests
# Cookers in isolation and defers cook_all/Driver by design (needs a
# project::Project tree). This pins the driver end to end: first cook
# bakes + writes the hash manifest, an unchanged cook skips everything,
# editing one source re-cooks only it, a deleted cooked output re-cooks
# on a manifest hit (fs::exists clause), force=true rebuilds
# unconditionally, and the manifest round-trips on disk — re-cooks of
# unchanged sources byte-identical. A regression here either re-cooks
# the whole project every save or ships STALE cooked bytes while every
# unit suite stays green. cardinal::cook brings import/asset/pack
# transitively; project supplies the tree. Pure CPU, deterministic,
# headless (skips with no temp dir). Exit 0 = pass.
add_executable(Cardinal_Test_CookIncremental main.cpp)
target_link_libraries(Cardinal_Test_CookIncremental PRIVATE
    cardinal::cook
    cardinal::project
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_CookIncremental PROPERTIES FOLDER tests)

add_test(NAME cook_incremental COMMAND Cardinal_Test_CookIncremental)
