// =============================================================================
// cardinal_clustering.hlsli — meshlet / cluster culling helpers (GPU side).
//
// Mirrors cardinal::render::geo on the CPU. Designed to be invoked from a
// compute or amplification shader once we wire GPU-driven culling. For now
// the engine drives clustering on the CPU; these helpers are the reference
// implementation the GPU pass will use when it lands.
//
// All functions take the same inputs as the C++ side so a unit test can
// trivially compare CPU vs. GPU results.
// =============================================================================

#ifndef CARDINAL_CLUSTERING_HLSLI
#define CARDINAL_CLUSTERING_HLSLI

// Per-meshlet bounds — must match cardinal::render::geo::ClusterBounds in
// memory layout when we go GPU-driven. Pad to 16-byte alignment so the
// struct is StructuredBuffer-friendly.
struct CardinalClusterBounds {
    float3 sphere_center;
    float  sphere_radius;
    float3 cone_apex;
    float  _pad0;
    float3 cone_axis;
    float  cone_angle_cos;
};

// Frustum — six outward-pointing planes (x, y, z, d) where the visible
// half-space is dot(plane.xyz, p) + plane.w >= 0.
struct CardinalFrustum {
    float4 planes[6];
};

bool cardinal_sphere_inside_frustum(CardinalFrustum f,
                                    float3 center, float radius)
{
    [unroll] for (int i = 0; i < 6; ++i) {
        float d = dot(f.planes[i].xyz, center) + f.planes[i].w;
        if (d < -radius) return false;
    }
    return true;
}

bool cardinal_cluster_passes_backface(CardinalClusterBounds b, float3 camera_pos) {
    // Wide-open cone — never reject.
    if (b.cone_angle_cos <= -0.9999f) return true;

    float3 to_cluster = b.sphere_center - camera_pos;
    float  l2         = dot(to_cluster, to_cluster);
    if (l2 < 1e-6f) return true;
    to_cluster *= rsqrt(l2);

    float d = dot(to_cluster, b.cone_axis);
    return d < b.cone_angle_cos;
}

// One-shot — call this from the amplification shader's payload thread to
// emit ONE meshlet per surviving lane.
bool cardinal_cluster_visible(CardinalClusterBounds b,
                              CardinalFrustum f,
                              float3 camera_pos)
{
    if (!cardinal_sphere_inside_frustum(f, b.sphere_center, b.sphere_radius))
        return false;
    return cardinal_cluster_passes_backface(b, camera_pos);
}

// LOD selection — coarse / fine / cull based on distance.
//   0 = full detail
//   1 = half (use a coarser cluster set)
//   2 = quarter
//   3 = cull entirely
uint cardinal_cluster_lod(CardinalClusterBounds b,
                          float3 camera_pos,
                          float  d_full,
                          float  d_half,
                          float  d_quarter)
{
    float d = length(b.sphere_center - camera_pos);
    if (d < d_full)    return 0;
    if (d < d_half)    return 1;
    if (d < d_quarter) return 2;
    return 3;
}

#endif  // CARDINAL_CLUSTERING_HLSLI
