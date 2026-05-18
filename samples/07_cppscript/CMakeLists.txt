add_executable(Cardinal_System_CppScript main.cpp)
target_link_libraries(Cardinal_System_CppScript PRIVATE
    cardinal::core cardinal::cppscript cardinal::plugin
    cardinal::compile_flags)
set_target_properties(Cardinal_System_CppScript PROPERTIES FOLDER samples)
