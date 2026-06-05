# Cardinal — core kit regression suite. Exercises every primitive in
# cardinal::core's recently-grouped subdirs (sync / container / clock / os /
# string / alloc / service): InterLock + ThreadLock + Access + lock guards,
# fixed-capacity stack strings, concurrent / waitable / circular / priority
# queues, Time / CpuTime / CpuUsage, Stopwatch + RepeatableTimer, Directory +
# File round-trip, worker Thread + cooperative stop, SehManager + minidump.
add_executable(Cardinal_Test_CoreKit main.cpp)
target_link_libraries(Cardinal_Test_CoreKit PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_CoreKit PROPERTIES FOLDER tests)

# Windows: SetThreadDescription lives in Kernel32 (already linked) but
# GetFileVersionInfo + VerQueryValue come from version.lib; dbghelp.dll is
# loaded dynamically by seh.cpp so no static lib dep there.
if(WIN32)
    target_link_libraries(Cardinal_Test_CoreKit PRIVATE version)
endif()

add_test(NAME core_kit COMMAND Cardinal_Test_CoreKit)
