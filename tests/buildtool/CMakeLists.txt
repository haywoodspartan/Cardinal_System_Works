# BuildCookRun pipeline (cardinal::buildtool) regression suite — drives the
# Build -> Cook -> Pack -> Archive orchestration end-to-end headless and asserts
# the staged shippable bundle on disk. Build stage generates only (no engine
# compile). Disk round-trips run through a unique temp dir. Exit 0 = all pass.
add_executable(Cardinal_Test_BuildTool main.cpp)
target_link_libraries(Cardinal_Test_BuildTool PRIVATE
    cardinal::buildtool
    cardinal::project
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_BuildTool PROPERTIES FOLDER tests)

add_test(NAME buildtool COMMAND Cardinal_Test_BuildTool)
