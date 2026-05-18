# cardinal::vgeom — virtualized geometry.
#
# Cooks a source mesh into a cluster hierarchy at import time and picks a
# per-frame "cut" through that hierarchy at runtime. The renderer transforms
# only the cut, so a 10M-tri mesh can render at the same cost as a 100K-tri
# one when the camera is far away — same visible LOD, fraction of the work.
#
# Toggleable per-mesh (vgeom::set_enabled). Falls through to the existing
# direct-vertex-buffer path when off.

set(CARDINAL_VGEOM_HEADERS
    include/cardinal/vgeom/vgeom.hpp
    include/cardinal/vgeom/cluster.hpp
)
set(CARDINAL_VGEOM_SOURCES
    src/cluster.cpp
    src/cook.cpp
    src/select.cpp
)

add_library(cardinal_vgeom STATIC ${CARDINAL_VGEOM_SOURCES} ${CARDINAL_VGEOM_HEADERS})
add_library(cardinal::vgeom ALIAS cardinal_vgeom)

target_include_directories(cardinal_vgeom PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# Concrete (core) for types/log; scene for Vertex/Mesh/Mat4.
target_link_libraries(cardinal_vgeom
    PUBLIC  cardinal::core cardinal::scene cardinal::compile_flags)

set_target_properties(cardinal_vgeom PROPERTIES FOLDER runtime)
