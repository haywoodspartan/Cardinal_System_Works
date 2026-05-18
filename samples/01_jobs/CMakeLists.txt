add_executable(Cardinal_System_Jobs main.cpp)
target_link_libraries(Cardinal_System_Jobs PRIVATE cardinal::core cardinal::compile_flags)
set_target_properties(Cardinal_System_Jobs PROPERTIES FOLDER samples)
