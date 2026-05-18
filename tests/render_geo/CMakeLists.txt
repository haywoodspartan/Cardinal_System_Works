# Deterministic meshlet / cluster-culling regression suite.
# render::geo is the Phase-6 virtualised-geometry baseline: build_meshlets
# greedily slices an indexed mesh into <=64-vert / <=64-prim clusters with
# a bounding sphere + backface cone, frustum_from_vp does Gribb-Hartmann
# plane extraction, and sphere_inside_frustum / cluster_passes_backface /
# cluster_lod are the CPU reference culling impls that a GPU compute pass
# must match. All of it is pure float math with NO RHI / GPU dependency,
# so the partition split rules (vertex cap, prim cap, bad-index skip,
# opts clamp), the contiguous index layout, the centroid sphere + min-dot
# cone (incl. the degenerate -> wide-open fallback), the identity-VP
# plane set, the sphere/plane and cone-rejection boundaries and the
# distance-LOD ramp are all exactly pinned. Exit 0 = all pass.
add_executable(Cardinal_Test_RenderGeo main.cpp)
target_link_libraries(Cardinal_Test_RenderGeo PRIVATE
    cardinal::render
    cardinal::core
    cardinal::compile_flags)
set_target_properties(Cardinal_Test_RenderGeo PROPERTIES FOLDER tests)

add_test(NAME render_geo COMMAND Cardinal_Test_RenderGeo)
