# Hang-watchdog regression: a silent heartbeat must fire ONE dump; a poked
# heartbeat must never fire. Time-based by nature (~5 s wall), Windows-real
# (minidump written + verified on disk), stub-tolerant elsewhere.
add_executable(Cardinal_Test_CrashWatchdog main.cpp)
target_link_libraries(Cardinal_Test_CrashWatchdog PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_CrashWatchdog PROPERTIES FOLDER tests)

add_test(NAME crash_watchdog COMMAND Cardinal_Test_CrashWatchdog)
