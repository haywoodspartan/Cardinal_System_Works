# Cardinal Asset Database — the asset pipeline's queryable index + dependency
# graph (Cardinal's Asset Registry). Built from a cook pass; sits above cook +
# asset (reads cooked material deps via the runtime Registry).
add_library(cardinal_assetdb STATIC
    src/assetdb.cpp
    include/cardinal/assetdb/assetdb.hpp)
add_library(cardinal::assetdb ALIAS cardinal_assetdb)

target_include_directories(cardinal_assetdb PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cardinal_assetdb
    PUBLIC  cardinal::core cardinal::cook cardinal::compile_flags
    PRIVATE cardinal::asset)

set_target_properties(cardinal_assetdb PROPERTIES FOLDER runtime)
