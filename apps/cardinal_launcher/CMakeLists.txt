if(NOT WIN32)
    return()    # POSIX launcher exe lands when WSI is wired in
endif()

# /SUBSYSTEM:WINDOWS — no console window when launched from the shell.
add_executable(cardinal_launcher_exe WIN32 main_windows.cpp)
add_executable(cardinal::launcher_exe ALIAS cardinal_launcher_exe)

target_link_libraries(cardinal_launcher_exe PRIVATE
    cardinal::launcher
    cardinal::core
    cardinal::compile_flags
    Shell32
)

set_target_properties(cardinal_launcher_exe PROPERTIES
    FOLDER apps
    OUTPUT_NAME "cardinal_launcher"   # final exe is cardinal_launcher.exe
)

cardinal_tp_deploy_runtimes(cardinal_launcher_exe)
