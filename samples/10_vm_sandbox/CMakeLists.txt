if(NOT WIN32)
    return()  # capability-sandbox VM host rides the Windows subprocess runner
endif()

add_executable(Cardinal_System_VmSandbox main.cpp)
target_link_libraries(Cardinal_System_VmSandbox PRIVATE
    cardinal::core cardinal::vm cardinal::sandbox
    cardinal::compile_flags)
set_target_properties(Cardinal_System_VmSandbox PROPERTIES FOLDER samples)
