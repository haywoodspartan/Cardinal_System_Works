# Deterministic virtual-texturing core regression suite. TileKey is the
# universal 64-bit currency (vt_id|mip|y|x bit-pack) behind every VT
# cache entry, request and page-table pointer; PageTable is the
# sampling-side residency state machine (lookup + mark_pending/resident/
# evicted/failed with lazy per-mip allocation and relaxed-atomic
# resident/pending counters). A regression silently corrupts texture
# streaming: aliased tile keys (wrong texture sampled), a mis-counted
# pool (VRAM leak or premature evict), or a lost status transition. The
# bit-pack is constexpr, so its exact layout is pinned at COMPILE TIME
# via static_assert (drift fails the build); the rest is pure + single-
# writer deterministic. Exit 0 = all pass.
add_executable(Cardinal_Test_VT main.cpp)
target_link_libraries(Cardinal_Test_VT PRIVATE
    cardinal::vt
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_VT PROPERTIES FOLDER tests)

add_test(NAME vt COMMAND Cardinal_Test_VT)
