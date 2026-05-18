# Phase-0 Foundation-vocabulary regression suite. The FOUNDATION RULE says
# non-core code must use the cardinal:: alias, never a std header directly.
# This suite is the contract for the freshly-added cardinal::core std-wrapper
# vocabulary (containers / algorithm / utility / traits / limits / cmath /
# cstring / bit / charconv / atomic / thread / chrono / filesystem /
# string_view) reached through the single umbrella <cardinal/core/std.hpp>.
# It compiles + exercises every alias so the umbrella stays free of
# ambiguities and the migration phases have a green baseline. Exit 0 = pass.
add_executable(Cardinal_Test_FoundationStd main.cpp)
target_link_libraries(Cardinal_Test_FoundationStd PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_FoundationStd PROPERTIES FOLDER tests)

add_test(NAME foundation_std COMMAND Cardinal_Test_FoundationStd)
