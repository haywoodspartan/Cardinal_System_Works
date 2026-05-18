# Deterministic CPK archive regression suite. The .cpk format is the
# client-distribution container — a regression here ships corrupt or
# unreadable game packs to players. Covers the Builder → Archive
# round-trip: byte-exact payload fidelity (incl. all-256 bytes, an
# embedded "CPK1" magic, empty + ~100 KB blobs), the on-disk FNV-1a64
# integrity field, the contiguous header→payload→index layout, the
# last-wins duplicate-key rule, and the open/read failure paths.
# Synchronous path only — load_async() needs the JobSystem + adds
# thread nondeterminism (separate concern). distribute() needs a cooked
# tree and belongs in a cook+pack integration arc. cardinal::pack
# brings cook+core transitively; /W4 /WX via compile_flags.
add_executable(Cardinal_Test_Pack main.cpp)
target_link_libraries(Cardinal_Test_Pack PRIVATE
    cardinal::pack
    cardinal::cook
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_Pack PROPERTIES FOLDER tests)

add_test(NAME pack COMMAND Cardinal_Test_Pack)
