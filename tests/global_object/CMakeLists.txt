# Cardinal — GlobalObjectManager + shared_object regression suite.
# Covers type-keyed + string-keyed register/get/unregister, concurrent
# registration safety, shared_object intern semantics (same key → same
# shared_ptr), weak_ptr auto-eviction when last ref drops, and the
# concurrent factory-race path.
add_executable(Cardinal_Test_GlobalObject main.cpp)
target_link_libraries(Cardinal_Test_GlobalObject PRIVATE
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_GlobalObject PROPERTIES FOLDER tests)

add_test(NAME global_object COMMAND Cardinal_Test_GlobalObject)
