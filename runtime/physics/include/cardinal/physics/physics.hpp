#pragma once

// =============================================================================
// Cardinal — Physics core.
//
// Minimal-but-real impulse-based rigid body physics. Three-shape narrow phase
// (Sphere / Box / Plane), uniform-grid broad phase, semi-implicit Euler
// integrator, fixed-timestep stepping with an accumulator so simulation
// determinism is independent of frame rate.
//
// Architecture:
//
//   World          — owns Bodies, runs step() at a fixed dt, dispatches contacts
//   Body           — Static / Kinematic / Dynamic; mass, velocity, transform
//   Collider       — { Sphere(r), Box(half_extents), Plane(n,d) }
//   Material       — restitution + friction
//   ContactEvent   — populated each step; cleared at the next step start
//
// The world step is:
//   1. integrate forces  — gravity → linear velocity, semi-implicit Euler
//   2. broad phase       — uniform grid, emits candidate AABB-pair set
//   3. narrow phase      — exact shape-shape tests; emits Contact records
//   4. solve             — sequential impulses (N iterations) for restitution
//                          and Coulomb friction
//   5. integrate motion  — velocity → position; clamp tiny velocities
//
// Goals: fast enough to step 1000 dynamic bodies a frame on the editor's
// budget, deterministic given the same input, no third-party deps.
// =============================================================================

#include <cardinal/core/handle.hpp>
#include <cardinal/core/math.hpp>
#include <cardinal/core/spatial.hpp>
#include <cardinal/core/types.hpp>        // function/memory/string
#include <cardinal/core/containers.hpp>   // cardinal::vector

namespace cardinal::physics {

// Canonical math + spatial types — single definitions live in
// cardinal::core. No physics::Vec3 / Quat / Mat3 / AABB / Ray duplicates;
// every cross-module call gets the same memory layout + semantics.
using Vec3 = cardinal::core::Vec3;
using Quat = cardinal::core::Quat;
using Mat3 = cardinal::core::Mat3;
using AABB = cardinal::core::AABB;
using Ray  = cardinal::core::Ray;

// Re-export the free vector/quat functions so unqualified `dot(a,b)`,
// `cross(a,b)`, `length(v)`, `normalize(v)`, `rotate(q,v)`, `axis_angle`,
// `integrate`, `quat_to_mat3` keep working inside namespace
// cardinal::physics.
using cardinal::core::dot;
using cardinal::core::cross;
using cardinal::core::length;
using cardinal::core::normalize;
using cardinal::core::rotate;
using cardinal::core::axis_angle;
using cardinal::core::integrate;
using cardinal::core::quat_to_mat3;

// Shapes.
enum class ShapeKind : u32 { Sphere = 0, Box, Plane, Capsule };

struct ShapeSphere  { f32 radius{0.5f}; };
struct ShapeBox     { Vec3 half_extents{0.5f, 0.5f, 0.5f}; };
struct ShapePlane   { Vec3 normal{0, 1, 0}; f32 d{0.0f}; };  // n·x + d = 0
// Capsule = swept sphere along the body's local Y axis. Total height
// = 2*(half_height + radius); the cylindrical middle is 2*half_height long.
struct ShapeCapsule { f32 radius{0.4f}; f32 half_height{0.5f}; };

struct Collider {
    ShapeKind kind{ShapeKind::Sphere};
    union {
        ShapeSphere  sphere;
        ShapeBox     box;
        ShapePlane   plane;
        ShapeCapsule capsule;
    };
    Collider() noexcept : kind(ShapeKind::Sphere), sphere{} {}
    static Collider make_sphere (f32 r) noexcept;
    static Collider make_box    (Vec3 he) noexcept;
    static Collider make_plane  (Vec3 n, f32 d) noexcept;
    static Collider make_capsule(f32 r, f32 half_h) noexcept;
};

struct Material {
    f32 restitution{0.2f};   // 0 = inelastic, 1 = elastic
    f32 friction   {0.5f};   // Coulomb (combined as μ_a * μ_b)
};

enum class BodyType : u32 {
    Static = 0,    // never moves; infinite mass; gravity does nothing
    Kinematic,     // user-driven; velocity respected, no force integration
    Dynamic,       // full physics; gravity + impulses apply
};

// One simulated body. Tag-typed handle prevents EntityId / PhysicsBodyId mixups.
struct BodyTag {};
using BodyHandle = cardinal::core::Handle<BodyTag>;

struct BodyDesc {
    BodyType  type{BodyType::Dynamic};
    Vec3      position{0, 0, 0};
    Vec3      velocity{0, 0, 0};
    f32       mass{1.0f};            // ignored for Static/Kinematic
    Collider  collider{};
    Material  material{};
    bool      gravity{true};         // disable per-body if e.g. floating debris
    // Collision filtering. A pair {A,B} interacts iff
    //   (A.mask & B.layer) != 0  ||  (B.mask & A.layer) != 0
    // and a body is hit by a raycast/overlap/sphere-cast iff
    //   (body.layer & query_layer_mask) != 0.
    // Both reduce to a bitwise-AND, so a body whose `layer` is 0 is on
    // NO layer: it ANDs to zero against everything and therefore collides
    // with nothing and is hit by no ray/overlap. The default is bit 0
    // (1u) so out-of-the-box bodies collide and raycast as expected;
    // assign distinct layer bits and/or narrow `mask` to opt into
    // filtering. `mask` defaults to all-ones (collide with every layer).
    u32       layer{1u};
    u32       mask {0xFFFFFFFFu};
    // Sensor / trigger volumes don't get pushed apart and don't apply
    // impulses, but they DO emit ContactEvents through the world's
    // contact callback. Useful for pickups, area triggers, kill volumes,
    // line-of-sight tests, etc.
    bool      is_sensor{false};

    // ---- Angular dynamics ----
    Quat      orientation{Quat::identity()};
    Vec3      angular_velocity{0, 0, 0};
    // Linear / angular damping coefficients applied each substep:
    //   v *= (1 - linear_damping  * dt)
    //   w *= (1 - angular_damping * dt)
    // Both clamped to [0, 1]. Use small values like 0.1 for slight drag.
    f32       linear_damping{0.0f};
    f32       angular_damping{0.05f};

    // Sleep thresholds. When BOTH linear-velocity² < linear_sleep_threshold²
    // AND angular-velocity² < angular_sleep_threshold² for `sleep_time_seconds`
    // contiguous seconds, the body is put to sleep — solver + integrator
    // skip it until a contact / impulse / force wakes it up.
    bool      can_sleep{true};
    f32       linear_sleep_threshold{0.10f};
    f32       angular_sleep_threshold{0.20f};
    f32       sleep_time_seconds{0.5f};
};

struct ContactEvent {
    BodyHandle a;
    BodyHandle b;
    Vec3       point;          // world-space contact point
    Vec3       normal;         // points from a → b
    f32        penetration{0}; // positive = overlap depth
};

struct RayHit {
    BodyHandle body{};
    Vec3       point{};
    Vec3       normal{};
    f32        t{0.0f};        // ray parameter (distance along ray.direction)
    bool       hit{false};
    explicit operator bool() const noexcept { return hit; }
};

// World — top-level physics container.
class World {
public:
    static cardinal::unique_ptr<World> create();
    virtual ~World() = default;
    World(const World&)            = delete;
    World& operator=(const World&) = delete;

    // Globals.
    virtual void set_gravity(Vec3 g) noexcept = 0;
    virtual Vec3 gravity() const noexcept    = 0;
    virtual void set_fixed_timestep(f32 dt) noexcept = 0;  // default 1/120
    virtual f32  fixed_timestep()  const noexcept    = 0;
    virtual void set_solver_iterations(u32 n) noexcept = 0;  // default 4
    virtual u32  solver_iterations() const noexcept    = 0;

    // Body lifecycle.
    virtual BodyHandle create_body(const BodyDesc& desc)   = 0;
    virtual bool       destroy_body(BodyHandle h)          = 0;
    virtual usize      body_count() const noexcept         = 0;

    // Per-body mutators (return false on stale handle).
    virtual bool set_position (BodyHandle, Vec3) = 0;
    virtual bool set_velocity (BodyHandle, Vec3) = 0;
    virtual bool apply_impulse(BodyHandle, Vec3) = 0;       // instantaneous Δv = J/m
    virtual bool apply_force  (BodyHandle, Vec3) = 0;       // accumulated through next step
    virtual bool teleport     (BodyHandle, Vec3 pos, Vec3 vel) = 0;

    // Per-body angular mutators.
    virtual bool set_orientation     (BodyHandle, Quat) = 0;
    virtual bool set_angular_velocity(BodyHandle, Vec3) = 0;
    virtual bool apply_torque        (BodyHandle, Vec3 tau) = 0;       // accumulated
    virtual bool apply_angular_impulse(BodyHandle, Vec3 j_ang) = 0;    // instantaneous Δω
    // Apply a force at a world-space point — produces both linear AND
    // angular impulse depending on the lever arm (point - centre_of_mass).
    virtual bool apply_force_at_point(BodyHandle, Vec3 force, Vec3 world_point) = 0;

    // Sleep control.
    virtual bool is_sleeping(BodyHandle) const = 0;
    virtual bool wake (BodyHandle) = 0;
    virtual bool sleep(BodyHandle) = 0;

    // Per-body queries.
    virtual Vec3     position    (BodyHandle) const = 0;
    virtual Vec3     velocity    (BodyHandle) const = 0;
    virtual Quat     orientation (BodyHandle) const = 0;
    virtual Vec3     angular_velocity(BodyHandle) const = 0;
    virtual BodyType type        (BodyHandle) const = 0;
    virtual AABB     world_aabb  (BodyHandle) const = 0;

    // Step the world by a wall-clock dt. Internally accumulates and runs
    // 0..K fixed-timestep substeps so simulation is deterministic.
    virtual void step(f32 wall_dt) = 0;

    // Per-step contact events (cleared at the next step start).
    virtual const cardinal::vector<ContactEvent>& last_contacts() const noexcept = 0;

    // Single-ray query against every body's collider. Returns the closest hit.
    virtual RayHit raycast(const Ray& r, f32 max_distance = 1e6f,
                           u32 layer_mask = 0xFFFFFFFFu) const = 0;

    // Multi-hit raycast — invoked once per body whose collider the ray
    // crosses. The callback returns false to stop iteration early. Returns
    // the total number of hits enumerated. Hits are NOT sorted — sort
    // yourself if you need front-to-back order.
    using RaycastCallback = cardinal::function<bool(const RayHit&)>;
    virtual usize raycast_all(const Ray& r, f32 max_distance,
                              u32 layer_mask,
                              const RaycastCallback& cb) const = 0;

    // Sphere cast — swept sphere from `r.origin` along `r.direction` for
    // `max_distance` units. Equivalent to a raycast inflated by `radius`.
    // Returns the closest body the swept sphere first contacts; degenerates
    // to raycast() when radius == 0.
    virtual RayHit sphere_cast(const Ray& r, f32 radius, f32 max_distance,
                               u32 layer_mask = 0xFFFFFFFFu) const = 0;

    // ---- Volume-overlap queries ---------------------------------------
    // Each invokes `cb` once per body whose AABB-then-precise test passes;
    // returning false from `cb` aborts iteration early. Returns the total
    // number of bodies enumerated. Layer-mask filtering follows the same
    // convention as raycast (`(body.layer & layer_mask) != 0`).
    using OverlapCallback = cardinal::function<bool(BodyHandle)>;
    virtual usize overlap_sphere(const Vec3& center, f32 radius,
                                 u32 layer_mask,
                                 const OverlapCallback& cb) const = 0;
    virtual usize overlap_aabb  (const AABB& volume,
                                 u32 layer_mask,
                                 const OverlapCallback& cb) const = 0;
    virtual usize overlap_box   (const Vec3& center, const Vec3& half_extents,
                                 u32 layer_mask,
                                 const OverlapCallback& cb) const = 0;

    // ---- Closest-point + distance queries -----------------------------
    // closest_point_on_body: nearest point on the body's collider to
    // `query`. Returns query itself if the handle is stale.
    virtual Vec3 closest_point_on_body(BodyHandle, const Vec3& query) const = 0;

    // Optional broadcast: invoked once per ContactEvent during step().
    using ContactCallback = cardinal::function<void(const ContactEvent&)>;
    virtual void set_contact_callback(ContactCallback cb) = 0;

    // ---- Joints ------------------------------------------------------
    // Distance joint: keeps |pos_a - pos_b| close to `rest_length`,
    // with optional softness (0 = hard, 1 = very soft). The joint applies
    // an impulse along the line connecting the two anchor points each
    // substep until the constraint is satisfied. Either body can be Static
    // (then it's a "rope to the world" — the dynamic body swings around it).
    struct JointTag {};
    using JointHandle = cardinal::core::Handle<JointTag>;
    struct DistanceJointDesc {
        BodyHandle a, b;
        Vec3       local_anchor_a{0, 0, 0};   // in body A's local frame
        Vec3       local_anchor_b{0, 0, 0};
        f32        rest_length{1.0f};
        f32        stiffness{1.0f};            // 0 = ignored, 1 = full impulse
        bool       break_on_overforce{false};
        f32        break_force{1e9f};
    };
    virtual JointHandle create_distance_joint(const DistanceJointDesc&) = 0;
    virtual bool        destroy_joint(JointHandle) = 0;
    virtual usize       joint_count() const noexcept = 0;

    // ---- Broad-phase spatial grid ------------------------------------
    // Replaces the O(N²) AABB pair test with a uniform-grid bucket. Cell
    // size should be ~2× the average body's largest extent. 0 = use
    // O(N²) (the old behaviour).
    virtual void set_broadphase_cell_size(f32 cell) noexcept = 0;
    virtual f32  broadphase_cell_size() const noexcept = 0;

    // ---- Time scale --------------------------------------------------
    // Multiplies every step()'s wall-clock dt before substepping. 1.0 =
    // realtime, 0.5 = bullet-time, 2.0 = double-speed, 0 = paused. Affects
    // ALL bodies + joints in this world. Per-body / per-region time
    // dilation is a follow-up.
    virtual void set_time_scale(f32 s) noexcept = 0;
    virtual f32  time_scale() const noexcept = 0;

    // ---- Force fields ------------------------------------------------
    // Region-based "physics bend" effects — gravity wells, anti-gravity
    // bubbles, drag zones, wind. Each tick the integrator queries every
    // dynamic body's position against every active field and applies the
    // resulting acceleration / force. Fields are AABB-bounded for cheap
    // broad rejection; the per-field force formula does the precise work.
    enum class FieldKind : u32 {
        UniformAccel,    // constant acceleration in `direction` (wind, anti-gravity)
        Radial,          // F = strength · normalize(centre - p) · falloff(r)
        Drag,            // F = -damping · v  (linear drag inside the volume)
        Vortex,          // F tangential to centre about `direction` axis
    };
    struct ForceField {
        FieldKind kind{FieldKind::UniformAccel};
        Vec3      centre{0, 0, 0};            // ignored for UniformAccel
        Vec3      direction{0, 1, 0};         // axis (Vortex) or accel dir (UniformAccel)
        f32       strength{1.0f};
        f32       radius{50.0f};              // 0 = unbounded (UniformAccel only)
        f32       falloff_exp{2.0f};          // r^(-falloff_exp), 0 = constant
        u32       layer_mask{0xFFFFFFFFu};    // affects bodies whose layer passes
    };
    struct FieldTag {};
    using FieldHandle = cardinal::core::Handle<FieldTag>;
    virtual FieldHandle add_force_field(const ForceField&) = 0;
    virtual bool        remove_force_field(FieldHandle) = 0;
    virtual usize       force_field_count() const noexcept = 0;

protected:
    World() = default;
};

// ============================================================================
// Standalone (no-World) collision queries.
//
// These take raw colliders + transforms — useful for gameplay code asking
// "would these intersect if placed here?" without first creating Bodies in
// the World. All routines use the same internal narrow-phase helpers as
// the World step, so results are bit-identical to what the simulator sees.
// ============================================================================

struct ShapeTransform {
    Vec3 position{0, 0, 0};
    // Rotation deferred — boxes are AABB-only in this version. Add a quat
    // here when oriented bounding boxes land.
};

// Returns true when the two colliders overlap. When `out_contact` is non-null,
// fills it with the (point, normal a→b, penetration) contact info.
bool test_overlap(const Collider& a, const ShapeTransform& at,
                  const Collider& b, const ShapeTransform& bt,
                  ContactEvent* out_contact = nullptr) noexcept;

// Closest point on the collider's surface (or interior if `query` is inside)
// to `query`. For a sphere: clamp to sphere shell; for a box: per-axis clamp;
// for a plane: orthogonal projection.
Vec3 closest_point_on_collider(const Collider&, const ShapeTransform&,
                               const Vec3& query) noexcept;

// length(query - closest_point). Returns 0 if `query` is inside the collider.
f32 distance_to_collider(const Collider&, const ShapeTransform&,
                         const Vec3& query) noexcept;

// AABB of a collider in world space (planes return a huge proxy).
AABB collider_world_aabb(const Collider&, const ShapeTransform&) noexcept;

}  // namespace cardinal::physics
