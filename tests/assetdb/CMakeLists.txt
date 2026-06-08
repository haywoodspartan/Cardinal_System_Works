# Asset Database (cardinal::assetdb) regression suite — builds an index from
# cook results + an in-memory material, exercises query/dependency API + the
# serialize/save-load round-trips. Headless. Exit 0 = pass.
add_executable(Cardinal_Test_AssetDb main.cpp)
target_link_libraries(Cardinal_Test_AssetDb PRIVATE
    cardinal::assetdb
    cardinal::asset
    cardinal::cook
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_AssetDb PROPERTIES FOLDER tests)

add_test(NAME assetdb COMMAND Cardinal_Test_AssetDb)
