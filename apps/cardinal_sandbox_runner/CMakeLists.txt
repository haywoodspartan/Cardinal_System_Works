if(NOT WIN32)
    # Subprocess sandbox runner is Windows-only today (POSIX path will use
    # posix_spawn + AF_UNIX socket — see sandbox.hpp). Skip on other OSes
    # so the build doesn't fail; cardinal::sandbox itself is cross-platform
    # but only its InProcess mode works without the runner.
    return()
endif()

add_executable(cardinal_sandbox_runner main.cpp)
target_link_libraries(cardinal_sandbox_runner PRIVATE
    cardinal::core cardinal::plugin cardinal::sandbox cardinal::vm
    cardinal::compile_flags)
set_target_properties(cardinal_sandbox_runner PROPERTIES FOLDER apps)
