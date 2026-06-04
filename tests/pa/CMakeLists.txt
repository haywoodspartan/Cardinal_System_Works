# Cardinal — pa::* regression suite. Exercises every modernised port of the
# Pearl Abyss PaLock / PaFile / PaDirectory / PaQueue / PaSeh / PaString /
# PaThread / PaTime / PaTimer surfaces under cardinal::core::pa::*.
add_executable(Cardinal_Test_Pa main.cpp)
target_link_libraries(Cardinal_Test_Pa PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Pa PROPERTIES FOLDER tests)

# Windows: SetThreadDescription lives in Kernel32 (already linked) but
# GetFileVersionInfo + VerQueryValue come from version.lib; dbghelp.dll is
# loaded dynamically by seh.cpp so no static lib dep there.
if(WIN32)
    target_link_libraries(Cardinal_Test_Pa PRIVATE version)
endif()

add_test(NAME pa COMMAND Cardinal_Test_Pa)
