if(NOT WIN32)
    return()  # subprocess sandbox is Windows-only today (see sandbox.cpp)
endif()

add_executable(Cardinal_System_Sandbox main.cpp)
target_link_libraries(Cardinal_System_Sandbox PRIVATE
    cardinal::core cardinal::cppscript cardinal::plugin cardinal::sandbox
    cardinal::compile_flags)
set_target_properties(Cardinal_System_Sandbox PROPERTIES FOLDER samples)
