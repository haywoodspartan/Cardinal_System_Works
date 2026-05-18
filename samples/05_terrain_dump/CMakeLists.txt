add_executable(Cardinal_System_TerrainDump main.cpp)
target_link_libraries(Cardinal_System_TerrainDump PRIVATE
    cardinal::core cardinal::scene cardinal::compile_flags)
set_target_properties(Cardinal_System_TerrainDump PROPERTIES FOLDER samples)
