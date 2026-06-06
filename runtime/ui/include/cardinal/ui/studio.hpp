#pragma once

// =============================================================================
// Cardinal Studio — Dear ImGui–driven editor UI host.
//
// The Studio takes ownership of the swapchain's "overlay" slot: it registers
// a callback with the swapchain that runs after scene rendering ends and
// before the present transition. Inside that callback Studio executes its
// own VkRenderPass against the current swapchain image and runs ImGui's
// draw data into it. Studio also exposes the swapchain's off-screen
// viewport image as an ImTextureID so it can be displayed inside an
// `ImGui::Begin("Viewport"); ImGui::Image(studio->viewport_texture(), sz);`
// panel — that's the editor viewport.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/scene/math.hpp>

#include <cardinal/core/containers.hpp>

namespace cardinal::rhi       { class Device;    class Swapchain; }
namespace cardinal::window    { class Window; }
namespace cardinal::scene     { class Scene;     enum class ViewMode : u32; }
namespace cardinal::script    { class Engine; }
namespace cardinal::render    { class Registry; }
namespace cardinal::cppscript { class Engine; }
namespace cardinal::budget    { class Broker; }
namespace cardinal::edit      { class EditorState; namespace brush { struct Brush; } }
namespace cardinal::vt        { class System; }
namespace cardinal::actor     { class World; }
namespace cardinal::sim       { class SimWorld; }
namespace cardinal::anim      { template <class T> struct Curve; }
namespace cardinal::audio     { class Engine; }
namespace cardinal::cine      { class Player; }
namespace cardinal::game      { class Game; }
namespace cardinal::sky       { class Sky; }
namespace cardinal::input     { class Manager; }
namespace cardinal::particles { class System; }
namespace cardinal::nav       { class Grid; }
namespace cardinal::project   { class Project; class RecentProjects; }
namespace cardinal::cook      { class CookerRegistry; }
namespace cardinal::shader    { class Compiler; }
namespace cardinal::io        { class Dispatcher; }
namespace cardinal::partition { class WorldPartition; }
namespace cardinal::level     { class LevelManager; struct HlodTree; }
namespace cardinal::mass      { class World; }
namespace cardinal::ai        { class PerceptionWorld; }
namespace cardinal::navmesh   { struct Mesh; }

namespace cardinal::ui {

class Studio {
public:
    static cardinal::unique_ptr<Studio> create(
        cardinal::rhi::Device&    device,
        cardinal::rhi::Swapchain& swapchain,
        cardinal::window::Window& window);

    virtual ~Studio() = default;

    Studio(const Studio&)            = delete;
    Studio& operator=(const Studio&) = delete;

    // Per-frame entry points, called by the application:
    //   begin_frame() — call AFTER swapchain->begin_frame() and AFTER recording
    //                   any 3D scene draws. Sets up ImGui::NewFrame.
    //   end_frame()   — call BEFORE swapchain->end_frame(). Just calls
    //                   ImGui::Render to finalise the draw lists; the actual
    //                   GPU recording happens later inside the swapchain's
    //                   overlay callback (which Studio registered).
    virtual void begin_frame() = 0;
    virtual void end_frame()   = 0;

    // Tell Studio that the swapchain was just resized. The app calls
    // swapchain->resize(w,h) first, then this. Vulkan rebuilds its per-image
    // framebuffers; D3D12 has no per-image state to rebuild.
    virtual void on_swapchain_resized() = 0;

    // Hot-swap onto a new (Device, Swapchain) pair — used by the engine's
    // backend-switch path (DX12 ↔ Vulkan at runtime, no relaunch).
    //
    // Releases every backend-bound resource Studio owns (ImGui's renderer
    // backend state, descriptor heaps / pools, viewport SRV bindings,
    // platform-window thunks), then re-runs the per-backend init against
    // the new device + swapchain. The ImGui CONTEXT (windows, dock layout,
    // .ini, font atlas data) is preserved across the swap so the user's
    // editor layout survives unchanged.
    //
    // Caller responsibilities:
    //   - GPU work on the OLD device must be drained before this is called
    //     (engine does vkDeviceWaitIdle / d3d12 fence wait first).
    //   - Tool-window registrations + console state survive in place.
    //
    // Returns false on init failure (in which case the editor is in an
    // unusable state — the engine should abort the swap and revert to the
    // previous backend, OR shut down).
    virtual bool reinit(rhi::Device& device, rhi::Swapchain& swapchain) = 0;

    // Drives ImGui's multi-viewport platform windows — call at the end of
    // each frame after end_frame(). When the user has dragged a panel
    // outside the main window, ImGui creates an OS-native child window for
    // it; this entrypoint pumps + renders those secondary windows. No-op
    // when ConfigFlags_ViewportsEnable is off.
    virtual void update_platform_windows() = 0;

    // Wipe every persisted ImGui window position / dock state. Useful when
    // the .ini gets stuck in a state where panels can't be reached (off-
    // screen positions from a previous monitor layout, broken docks, etc.).
    // The next frame re-runs the host's dockspace builder and snaps any
    // restored windows back onto the main monitor.
    virtual void reset_layout() = 0;

    // Maximize / restore the dock panel that currently has focus. ImGui
    // docked windows (Hierarchy, Inspector, Console, …) have no native
    // maximize button — only the viewport panels carry one in their
    // toolbar. This floats the focused panel over the full host work
    // area (covering everything else) and, on a second call, snaps it
    // back into the dock node it came from. Studio also binds Shift+F11
    // to this so it works without a host menu item; hosts can surface a
    // "Window -> Maximize Panel" menu entry that calls it too. No-op
    // when focus is on the dockhost / a popup / nothing.
    virtual void toggle_maximize_focused_panel() = 0;

    // Built-in panels. Call between begin_frame() and end_frame().
    //   draw_fps_overlay() — small floating top-left FPS / frame-time card
    //   draw_log_panel()   — windowed message stream sourced from the
    //                        engine-wide cardinal::log facility. Pass nullptr
    //                        to draw with a default title; pass a bool*
    //                        if the View menu should be able to toggle it.
    //   draw_stats_panel() — RHI capabilities + active render settings, plus
    //                        per-frame counters reported by the swapchain.
    virtual void draw_fps_overlay() = 0;
    virtual void draw_log_panel  (const char* title = "Log",   bool* p_open = nullptr) = 0;
    virtual void draw_stats_panel(const char* title = "Stats", bool* p_open = nullptr) = 0;

    // Live RAM / VRAM pressure + per-subsystem budget bars. Pass the host's
    // budget::Broker; nullptr renders a placeholder.
    virtual void draw_memory_panel(cardinal::budget::Broker* broker,
                                   const char* title = "Memory & Budgets",
                                   bool* p_open = nullptr) = 0;

    // Frame timing + per-phase EMA + per-worker busy bars.
    // Host hands the most recent end-of-frame ms (so the chart matches
    // what the user is actually waiting on).
    virtual void draw_profiler_panel(float frame_ms_now,
                                     const char* title = "Profiler",
                                     bool* p_open = nullptr) = 0;

    // Virtual texturing residency / cache / per-VT mip table.
    virtual void draw_vt_panel(cardinal::vt::System* sys,
                               const char* title = "Virtual Textures",
                               bool* p_open = nullptr) = 0;

    // ----- Actor / Simulation / Animation / Audio / Cinematic ----------
    virtual void draw_sim_bar_panel(cardinal::sim::SimWorld* sim,
                                    const char* title = "Simulation",
                                    bool* p_open = nullptr) = 0;
    virtual void draw_actor_outliner_panel(cardinal::actor::World* world,
                                           u32* selected_actor_id_inout,
                                           const char* title = "Actors",
                                           bool* p_open = nullptr) = 0;
    // Prefab Library — capture the selected actor into a reusable template
    // + stamp out independent instances. Studio owns the panel state
    // (in-progress name buffer); no host State object required.
    virtual void draw_prefab_panel(cardinal::actor::World* world,
                                   u32* selected_actor_id_inout,
                                   const char* title = "Prefabs",
                                   bool* p_open = nullptr) = 0;
    virtual void draw_curve_editor_panel(cardinal::anim::Curve<float>* curve,
                                         const char* title = "Curve Editor",
                                         bool* p_open = nullptr) = 0;
    virtual void draw_sequencer_panel(cardinal::cine::Player* player,
                                      const char* title = "Sequencer",
                                      bool* p_open = nullptr) = 0;
    virtual void draw_mixer_panel(cardinal::audio::Engine* engine,
                                  const char* title = "Mixer",
                                  bool* p_open = nullptr) = 0;

    // ----- Game lifecycle / Class picker / Sky -------------------------
    // Per-frame game logic (auto-checkpoint debounce + Play/Pause/Stop
    // hotkeys + PIE lifecycle). Call EVERY frame, unconditionally — NOT
    // gated by the Game panel's visibility — so undo checkpoints + hotkeys
    // keep working when the panel is closed or a viewport is maximized.
    virtual void update_game_logic(cardinal::game::Game* game) = 0;
    virtual void draw_game_bar_panel(cardinal::game::Game* game,
                                     const char* title = "Game",
                                     bool* p_open = nullptr) = 0;
    virtual void draw_class_picker_panel(cardinal::game::Game* game,
                                         u32* selected_actor_id_inout,
                                         const char* title = "Game Classes",
                                         bool* p_open = nullptr) = 0;
    virtual void draw_sky_panel(cardinal::sky::Sky* sky,
                                const char* title = "Sky / Time of Day",
                                bool* p_open = nullptr) = 0;

    // ----- Input / Save&Load / Particles / Navigation -------------------
    virtual void draw_input_panel(cardinal::input::Manager* mgr,
                                  const char* title = "Input",
                                  bool* p_open = nullptr) = 0;
    virtual void draw_save_load_panel(cardinal::game::Game* game,
                                      cardinal::sky::Sky*  sky,
                                      const char* title = "Save / Load",
                                      bool* p_open = nullptr) = 0;
    virtual void draw_particles_panel(cardinal::particles::System* sys,
                                      const char* title = "Particles",
                                      bool* p_open = nullptr) = 0;

    // Navigation panel state (start/goal cells + paint mode + cell pixel
    // size) is held by the host so the panel survives layout swaps.
    struct NavState {
        i32  start_x{1}, start_y{1};
        i32  goal_x{20}, goal_y{20};
        bool allow_diagonal{true};
        bool paint_blocked{true};
        f32  cell_pixels{16.0f};
    };
    virtual void draw_nav_panel(cardinal::nav::Grid* grid,
                                NavState* state,
                                const char* title = "Navigation",
                                bool* p_open = nullptr) = 0;

    // ----- Project / Cook & Pack / Shader compiler ---------------------
    struct ProjectAction {
        cardinal::shared_ptr<cardinal::project::Project> opened;   // non-null when Open/Create fired
        bool save_clicked{false};
        bool cook_clicked{false};
        bool pack_clicked{false};
        bool cook_and_pack_clicked{false};
    };
    virtual ProjectAction draw_project_panel(
        cardinal::shared_ptr<cardinal::project::Project>* current_project,
        cardinal::project::RecentProjects* recents,
        const char* title = "Project", bool* p_open = nullptr) = 0;

    // Cook & Pack — Studio owns the per-frame state internally so the host
    // doesn't have to manage cook::CookResult lists.
    virtual void draw_cook_pack_panel(
        cardinal::project::Project* project,
        cardinal::cook::CookerRegistry* registry,
        cardinal::shader::Compiler* shader_compiler,
        const char* title = "Cook & Pack", bool* p_open = nullptr) = 0;

    virtual void draw_shader_compiler_panel(
        cardinal::shader::Compiler* compiler,
        const char* title = "Shader Compiler", bool* p_open = nullptr) = 0;

    // ----- Combined "World Systems" tab panel --------------------------
    // Surfaces HAL info, IO dispatcher stats, World Partition cells,
    // Level placements, HLOD tree, Mass entities, AI sensors, Navmesh
    // stats — all in a single tabbed window so the View menu doesn't grow
    // linearly with every new low-level subsystem.
    struct WorldSystemsInputs {
        cardinal::io::Dispatcher*           io_dispatcher{nullptr};
        cardinal::partition::WorldPartition* partition{nullptr};
        cardinal::level::LevelManager*      level_manager{nullptr};
        const cardinal::level::HlodTree*    hlod_tree{nullptr};
        cardinal::mass::World*              mass_world{nullptr};
        cardinal::ai::PerceptionWorld*      ai_perception{nullptr};
        const cardinal::navmesh::Mesh*      navmesh{nullptr};
    };
    virtual void draw_world_systems_panel(
        const WorldSystemsInputs& inputs,
        const char* title = "World Systems",
        bool* p_open = nullptr) = 0;

    // ----- Editor mode toolbar / brush / mesh tools / texture tools ----
    //
    // Each panel binds against an externally-owned state object so the
    // host owns persistence + cross-frame mutation.
    virtual void draw_editor_modes_panel(cardinal::edit::EditorState* state,
                                         const char* title = "Modes",
                                         bool* p_open = nullptr) = 0;

    virtual void draw_brush_panel(cardinal::edit::brush::Brush* brush,
                                  const char* title = "Brush",
                                  bool* p_open = nullptr) = 0;

    // Mesh + texture tools panels — Studio owns the panel state internally,
    // hosts read the per-frame "request" snapshot to know what the user
    // asked for. Each clicked-* flag is true exactly the frame the button
    // fired; the host applies the action and the state resets next frame.

    enum class PrimitiveKind : u32 {
        Box, Sphere, Cylinder, Cone, Torus, Disk, Plane
    };
    struct MeshToolsRequest {
        // Generate primitive
        bool          generate{false};
        PrimitiveKind prim{PrimitiveKind::Box};
        float         size{1.0f};
        float         radius{1.0f};
        float         height{2.0f};
        float         minor{0.3f};
        u32           segments{24};

        // Edits to selected mesh
        bool          subdivide{false};
        u32           subdivide_levels{1};
        bool          mirror{false};
        u32           mirror_axis{0};
        bool          decimate{false};
        float         decimate_cell{0.25f};
        bool          smooth{false};
        u32           smooth_iters{2};
        float         smooth_lambda{0.5f};
        bool          recompute_normals{false};
        bool          recompute_smooth{true};
    };
    virtual MeshToolsRequest draw_mesh_tools_panel(
        bool selection_has_mesh,
        const char* title = "Mesh Tools",
        bool* p_open = nullptr) = 0;

    struct TextureToolsRequest {
        bool        export_now{false};
        cardinal::string export_path;     // populated when export_now is true
    };
    virtual TextureToolsRequest draw_texture_tools_panel(
        const char* title = "Texture Tools",
        bool* p_open = nullptr) = 0;

    // ----- C++ scripting / sandbox panel -------------------------------
    //
    // Exposes the cardinal::cppscript::Engine in a docked panel: compile +
    // load + unload + reload + watch a .cpp at runtime, see compile
    // diagnostics, see per-script sandbox status if the host has bound a
    // sandbox to the loaded plugin. Pass nullptr to suppress.
    using SandboxStatusLookup = cardinal::function<bool(const cardinal::string&,
                                                   void* /* sandbox::Status* */)>;
    virtual void draw_cppscript_panel(cardinal::cppscript::Engine* engine,
                                      const char* title = "Code Sandbox",
                                      bool*       p_open = nullptr,
                                      SandboxStatusLookup lookup = nullptr) = 0;

    // Returns an ImTextureID-castable handle for the swapchain's off-screen
    // viewport image. Use as the texture for ImGui::Image() in a viewport
    // panel. Returns nullptr if the swapchain has no viewport configured
    // (i.e. set_viewport_size was never called or set to 0,0).
    //
    // The single-arg form returns viewport 0 (legacy / single-viewport
    // hosts). The id-arg form returns the SRV for that specific viewport
    // when running multi-viewport — each editor panel passes its own id so
    // every panel samples its own RTT and can carry its own ViewMode.
    virtual void* viewport_texture() = 0;
    virtual void* viewport_texture(u32 viewport_id) = 0;

    // Per-frame snapshot of viewport interaction state. Populated by
    // draw_viewport_panel(); the host's main loop reads this AFTER
    // end_frame to do picking, click-to-place, etc. NDC is in [-1, +1]
    // with +Y up, matching standard projection conventions.
    struct ViewportPick {
        bool  hovered{false};       // mouse over a viewport panel this frame
        bool  click_left{false};    // LMB clicked (release-without-drag) this frame
        float ndc_x{0.0f};          // mouse X in panel-local NDC at click time
        float ndc_y{0.0f};
        // The panel the NDC/marquee was sampled in, and that panel's true
        // pixel aspect (width/height). The host MUST build its picking
        // projection with THIS aspect: a viewport dragged into its own OS
        // window (multi-viewport) has its own aspect, and pairing this
        // panel-local NDC with another panel's aspect skews the
        // unprojected ray — object placement lands offset. `aspect` is 0
        // when no panel was hovered (host should fall back).
        cardinal::u32 viewport_id{0};
        float         aspect{0.0f};
        bool  delete_pressed{false};// Delete / Backspace fired this frame
        // ---- Multi-select additions (back-compat: default false/0) -----
        // `additive` = Ctrl or Shift was held at click time → the host
        // should toggle/extend the selection set rather than replace it.
        bool  additive{false};
        // `marquee` = a left-drag box-select completed over empty space
        // this frame. The rect is panel-local NDC ([-1,+1], +Y up), with
        // (min_x,min_y) the bottom-left corner. The host tests each
        // entity's projected position/bounds against this rect and sets
        // the selection set accordingly (Studio stays scene-agnostic).
        bool  marquee{false};
        float marquee_min_x{0.0f}, marquee_min_y{0.0f};
        float marquee_max_x{0.0f}, marquee_max_y{0.0f};
        bool  marquee_additive{false};   // Ctrl/Shift held during the box
    };
    virtual ViewportPick last_viewport_pick() const noexcept = 0;

    // Last chunk under the mouse cursor while it was over a viewport with
    // the world-grid overlay enabled. Filled each frame by
    // draw_world_grid_overlay; empty (`valid` false) when the cursor isn't
    // over the grid plane this frame. Hosts use this for level-editor
    // affordances — drag-place into a specific chunk, "spawn here" right-
    // click action, snap-to-chunk constraints, etc.
    struct HoverChunk {
        bool valid{false};
        cardinal::i32 x{0};
        cardinal::i32 y{0};
        cardinal::i32 z{0};
    };
    virtual HoverChunk last_hover_chunk() const noexcept = 0;

    // ----- Translate gizmo -----------------------------------------------
    //
    // The Studio renders an Blender/UE5-style 3-axis translate handle on the
    // selected entity directly into the viewport panel via ImDrawList. The
    // host owns the camera + entity state, so each frame it tells Studio:
    //
    //   set_viewport_camera(view, proj)        — current camera matrices
    //   set_gizmo_target(world_pos, has_target)— where the gizmo lives
    //
    // …and Studio reports back through `last_gizmo_drag()` how much the
    // user dragged this frame. The host SHOULD assign `target_world`
    // directly to the entity's translation (absolute path) — that
    // eliminates per-frame float-add accumulation that otherwise shows
    // up as visible jitter for entities far from the world origin.
    //
    // `delta` is kept for back-compat but discouraged: float addition is
    // not associative, so `e.translation += delta` repeatedly compounds
    // round-off error scaled by |translation|. The user feels this as
    // jitter that gets WORSE the further the entity travels.
    //
    // Snap-to-grid: apply to `target_world` and assign the snapped value;
    // then call `set_gizmo_target(snapped, true)` so the gizmo stays
    // visually attached to the snapped entity instead of drifting away.
    // The gizmo has three modes — Translate (3 axes + 3 planes + a
    // screen-space free-move square), Rotate (3 axis rings), Scale (3
    // axis box-handles + a uniform centre box). W/E/R toggle them while
    // a viewport is hovered; the host can also force a mode.
    enum class GizmoMode : u32 { Translate = 0, Rotate = 1, Scale = 2 };

    virtual void set_viewport_camera(const cardinal::scene::Mat4& view,
                                     const cardinal::scene::Mat4& proj) = 0;
    // Position-only target — rotation/scale assumed identity. Rotate/
    // Scale gizmos need the full transform; prefer set_gizmo_transform.
    virtual void set_gizmo_target(const cardinal::scene::Vec3& world_pos,
                                  bool has_target) = 0;
    // Full transform target. `rotation_euler` is XYZ radians, matching
    // scene::Transform. Required for the Rotate/Scale gizmos to anchor
    // and to report meaningful absolute results.
    virtual void set_gizmo_transform(const cardinal::scene::Vec3& position,
                                     const cardinal::scene::Vec3& rotation_euler,
                                     const cardinal::scene::Vec3& scale,
                                     bool has_target) = 0;
    virtual void      set_gizmo_mode(GizmoMode m) noexcept = 0;
    virtual GizmoMode gizmo_mode() const noexcept = 0;
    // Gizmo-side snapping (so every host gets it right + the gizmo can
    // render snapped). step semantics by mode: Translate = world units,
    // Rotate = degrees, Scale = ratio increment. step <= 0 disables.
    virtual void set_gizmo_snap(bool enabled, float step) noexcept = 0;

    // ----- ImGuizmo widget toggle -------------------------------------
    // Studio ships with two gizmo renderers — Cardinal's hand-rolled
    // overlay and a vcpkg ImGuizmo (industry-standard, ships local/world
    // toggle + Universal-op + ViewManipulate orbit cube). Default is
    // ImGuizmo. The hand-rolled path stays in tree for A/B testing and
    // for hosts that prefer the Cardinal styling.
    virtual void set_use_imguizmo(bool enabled) noexcept = 0;
    virtual bool use_imguizmo() const noexcept = 0;
    // World-space vs object-local-space transform — ImGuizmo only.
    virtual void set_imguizmo_local_space(bool local) noexcept = 0;
    virtual bool imguizmo_local_space() const noexcept = 0;
    // Corner-cube camera orbiter widget (ViewManipulate) — ImGuizmo only.
    virtual void set_imguizmo_show_view_cube(bool show) noexcept = 0;

    struct GizmoDrag {
        bool                  active{false};
        GizmoMode             mode{GizmoMode::Translate};
        // --- Translate -------------------------------------------------
        // World-space delta since LAST frame. Kept for back-compat; new
        // code should prefer target_world (see comment above).
        cardinal::scene::Vec3 delta{0.0f, 0.0f, 0.0f};
        // Absolute world-space position the gizmo wants the entity at,
        // recomputed every frame from drag-start origin + axis projection
        // — no integration, no drift. Use this for assignment.
        cardinal::scene::Vec3 target_world{0.0f, 0.0f, 0.0f};
        // --- Rotate ----------------------------------------------------
        // Absolute target orientation (XYZ euler radians), recomputed
        // every frame as drag-origin-orientation composed with the
        // dragged axis-angle. Drift-free; assign directly.
        cardinal::scene::Vec3 target_rotation_euler{0.0f, 0.0f, 0.0f};
        // --- Scale -----------------------------------------------------
        // Absolute target scale, drag-origin scale × per-axis factor
        // (uniform handle scales all three). Assign directly.
        cardinal::scene::Vec3 target_scale{1.0f, 1.0f, 1.0f};
    };
    virtual GizmoDrag last_gizmo_drag() const noexcept = 0;

    // Gizmo drag-edge events for undo coalescing. The host watches the
    // release edge to push a single undo record at the end of a drag
    // rather than one per frame.
    virtual bool gizmo_drag_in_progress() const noexcept = 0;
    virtual bool gizmo_drag_release_consume() = 0;   // returns true once per release

    // ----- Multi-selection --------------------------------------------
    // Studio holds the selection set so the hierarchy + viewport agree.
    // Single-id call sites keep working unchanged: the legacy
    // draw_scene_hierarchy(scene,&id,…) / draw_inspector(scene,id,…)
    // mirror the id ⇄ a 1-element set. Hosts wanting true multi-select
    // use the vector overload and read selection() back. The gizmo
    // pivot (centroid) and the apply-to-all fan-out stay host-side so
    // Studio remains scene-agnostic — it only reports intent.
    virtual void set_selection(const u32* ids, cardinal::usize count) = 0;
    virtual const cardinal::vector<u32>& selection() const noexcept = 0;
    virtual void draw_scene_hierarchy(scene::Scene& scene,
                                      cardinal::vector<u32>* selection_inout,
                                      const char* title = "Hierarchy",
                                      bool* p_open = nullptr) = 0;

    // -------------------------------------------------------------------
    // Editor panels for the level-editor workflow.
    //
    // draw_viewport_panel(title, mode_inout) — a draggable / detachable
    //   panel that displays the current viewport texture, with a toolbar
    //   for picking the active rendering view-mode (Solid / Wireframe /
    //   Polygons / Heightmap / Normals / RTX preview). The chosen mode is
    //   read back through `mode_inout` so the caller drives the actual
    //   render pass — Studio is purely the UI surface.
    //
    // draw_scene_hierarchy(scene, selected_id_inout) — flat list of all
    //   entities with click-to-select semantics. selected_id_inout is the
    //   currently-selected entity id (0 = none).
    //
    // draw_inspector(scene, selected_id) — properties of the selected
    //   entity (Transform editor, mesh name, tint, visibility, delete).
    //
    // draw_asset_browser(root_path) — directory tree on the left, filtered
    //   grid of files on the right. Recognises common asset extensions
    //   (.gltf / .glb / .obj / .png / .jpg / .hlsl / .lua / etc.).
    //
    // draw_console_panel() — REPL surface; lines typed are dispatched via
    //   the user-supplied evaluator callback.
    // -------------------------------------------------------------------
    virtual void draw_viewport_panel(const char* title,
                                     scene::ViewMode* mode_inout,
                                     bool* p_open = nullptr) = 0;

    // Per-viewport FPS limit. Each `draw_viewport_panel` call site can
    // expose its own dropdown (Unlimited / 240 / 144 / 120 / 60 / 30 / 15)
    // backed by `*fps_limit_inout`. The host then forwards each
    // viewport's chosen value into a shared cardinal::core::FramePacer
    // so the actual main-loop sleep honours the strictest active cap.
    //
    // 0.0f means "Unlimited" — the dropdown sentinel value. Values <= 0
    // are clamped to 0. Pass nullptr to suppress the dropdown.
    virtual void draw_viewport_panel(const char* title,
                                     scene::ViewMode* mode_inout,
                                     float* fps_limit_inout,
                                     bool* p_open = nullptr) = 0;

    // Multi-viewport overload. Each editor viewport panel passes its
    // unique `viewport_id` (0..N-1) so it samples its own swapchain RTT
    // and the ViewMode dropdown is local to that panel. The host calls
    // rhi::Swapchain::set_active_viewport(id) + pipeline.render() per
    // panel, then this overload binds the matching SRV for ImGui::Image.
    //
    // The optional `maximized_inout` and `minimized_inout` flags drive the
    // toolbar's Maximize / Minimize buttons. Maximize is host-coordinated:
    // when set, the studio raises this panel above its dock and the host
    // typically hides the others. Minimize uses ImGui's window-collapse.
    // Pass nullptr to omit either button.
    virtual void draw_viewport_panel(const char* title,
                                     u32 viewport_id,
                                     scene::ViewMode* mode_inout,
                                     float* fps_limit_inout,
                                     bool* maximized_inout,
                                     bool* minimized_inout,
                                     bool* p_open = nullptr) = 0;
    virtual void draw_scene_hierarchy(scene::Scene& scene,
                                      u32* selected_id_inout,
                                      const char* title = "Hierarchy",
                                      bool* p_open = nullptr) = 0;
    virtual void draw_inspector(scene::Scene& scene, u32 selected_id,
                                const char* title = "Inspector",
                                bool* p_open = nullptr) = 0;
    virtual void draw_asset_browser(const char* root_path,
                                    const char* title = "Assets",
                                    bool* p_open = nullptr) = 0;

    // Asset palette — listing of every entry in scene::AssetCatalog,
    // grouped by category. Clicking the "Place" button on an entry calls
    // `on_pick(asset_id)`; the host typically flips its tool-state into
    // place-mode and remembers the id so the next viewport click instantiates.
    // The optional `on_spawn_at_camera` callback is wired to the
    // right-click context menu (Spawn at camera / Spawn at origin).
    using AssetPickFn  = cardinal::function<void(const char* asset_id)>;
    using AssetSpawnAt = cardinal::function<void(const char* asset_id, int where /* 0=camera, 1=origin */)>;
    virtual void draw_asset_palette(const AssetPickFn& on_pick,
                                    const char* current_asset_id = nullptr,
                                    const char* title = "Asset Palette",
                                    bool* p_open = nullptr) = 0;
    virtual void draw_asset_palette(const AssetPickFn& on_pick,
                                    const AssetSpawnAt& on_spawn_at,
                                    const char* current_asset_id,
                                    const char* title = "Asset Palette",
                                    bool* p_open = nullptr) = 0;

    // ----- Game-engine toolbar + context menus -------------------------
    //
    // draw_main_toolbar() — horizontal bar typically docked at the top of
    //   the editor: tool radio (Select / Place), snap toggle + cell size,
    //   undo / redo buttons (gated by what the UndoStack reports), play /
    //   pause / step buttons (engine-mode placeholder for now), and a
    //   "frame selection" camera-focus button.
    struct ToolbarState {
        // Tool radio backing — host owns the int and feeds it back next frame.
        int   tool{0};                 // 0=Select, 1=Place (anything else: app-defined)
        bool  snap_enabled{false};
        float snap_step{1.0f};

        // Engine play-mode flags — purely UI today (no actual sim split yet).
        // Set by buttons; host decides what to do with them.
        bool  play{false};
        bool  paused{false};
        bool  step_once{false};        // true exactly the frame Step is clicked

        // Camera commands the host can act on.
        bool  focus_selection{false};
        bool  reset_camera{false};
    };
    struct UndoButtons {
        bool can_undo{false};
        bool can_redo{false};
        const char* undo_label{""};   // for tooltip
        const char* redo_label{""};
        bool undo_clicked{false};      // true exactly the frame the button fires
        bool redo_clicked{false};
    };
    virtual void draw_main_toolbar(ToolbarState* state,
                                   UndoButtons* undo,
                                   const char* title = "Toolbar",
                                   bool* p_open = nullptr) = 0;

    // Context menu builders — return true if the menu was opened this frame
    // (so the host knows to wire the respective action callbacks).
    struct ViewportCtxAction {
        bool delete_selected{false};
        bool duplicate_selected{false};
        bool focus_selected{false};
        bool spawn_active_asset_at_cursor{false};
        bool clear_selection{false};
    };
    // Reads the per-frame context-menu action snapshot. Studio renders the
    // viewport context menu inside draw_viewport_panel (because the popup
    // must be inside that window's Begin/End scope); the host calls this
    // accessor AFTER end_frame to find out what the user picked.
    virtual ViewportCtxAction last_viewport_ctx_action() const noexcept = 0;
    // Legacy direct-render path for hosts that DO have an open ImGui
    // window when they call this. Sets the snapshot too.
    virtual void draw_viewport_context_menu(ViewportCtxAction* out) = 0;

    struct HierarchyCtxAction {
        bool rename{false};
        bool duplicate{false};
        bool delete_entity{false};
        bool focus_camera{false};
    };
    virtual void draw_hierarchy_context_menu(HierarchyCtxAction* out) = 0;

    // ----- Chunked-world panel + viewport grid overlay -----
    //
    // draw_world_panel exposes the editor controls for the streaming
    // system: chunk size, world extent, render distance + a live readout
    // of camera chunk + active chunk count. State is host-owned (mirrors
    // a cardinal::world::WorldGrid) so this panel is just a view.
    struct WorldPanelState {
        // Mirrors cardinal::world::WorldGridDesc — the host writes the
        // current values in; mutated values get written back when the user
        // moves a slider. Host then forwards to the equivalent setter.
        // Per-axis extents (X / Y / Z) + per-axis render distances allow
        // wide horizontal worlds with shallow vertical extent (typical) or
        // pure-cube voxel worlds (set Y == XZ to taste).
        f32 chunk_size{32.0f};
        i32 render_distance_xz{8};
        i32 render_distance_y{4};
        i32 extent_x{1024};
        i32 extent_y{64};
        i32 extent_z{1024};
        // Camera readouts (3D) + active chunk count (host fills these).
        i32 camera_chunk_x{0};
        i32 camera_chunk_y{0};
        i32 camera_chunk_z{0};
        i32 active_chunk_count{0};
        // Visualisation toggles.
        bool show_grid_overlay{true};
        bool cull_outside_render_distance{true};
        // Output: true the frame the user clicks "Force evict".
        bool evict_all_clicked{false};
    };
    virtual void draw_world_panel(WorldPanelState* state,
                                  const char* title = "World",
                                  bool* p_open = nullptr) = 0;

    // Overlay state pushed by the host each frame so the viewport panel
    // can render the chunk grid + render-distance highlight + current
    // camera-chunk highlight on the Y=0 plane (and a y-axis tick).
    struct WorldGridOverlay {
        bool show{false};
        f32  chunk_size{32.0f};
        i32  render_distance_xz{8};
        i32  render_distance_y{4};
        i32  camera_chunk_x{0};
        i32  camera_chunk_y{0};
        i32  camera_chunk_z{0};
    };
    virtual void set_world_grid_overlay(const WorldGridOverlay&) = 0;

    // Terrain Grid spawn modal — full world-engine workflow:
    //
    //   * Pick a TerrainProfile from the process-wide library
    //   * Edit the active profile's layers in place (add / remove / reorder)
    //   * Save the entire library to disk; load a library from disk
    //   * Pick the grid extent + per-chunk resolution / chunk size
    //   * Hit Generate → host receives a TerrainSpawnRequest with
    //     `profile_name` set, looks the profile up, and calls
    //     scene::spawn_terrain_grid_profile.
    //
    // Call this every frame; the modal opens when `open_now` flips to true.
    // `request_out->requested` is true exactly on the frame the user
    // hits Generate; false otherwise.
    struct TerrainSpawnRequest {
        bool        requested{false};
        cardinal::string profile_name;          // name of profile in library to use
        u32         resolution{64};
        float       chunk_size{32.0f};
        int         chunks_x{4};
        int         chunks_z{4};
    };
    virtual void draw_terrain_modal(bool open_now,
                                    TerrainSpawnRequest* request_out) = 0;

    using ConsoleEval = cardinal::function<cardinal::string(const char* line)>;
    virtual void draw_console_panel(const ConsoleEval& eval,
                                    const char* title = "Console",
                                    bool* p_open = nullptr) = 0;

    // ----- Tool-window registration ------------------------------------
    //
    // Apps can register additional dockable panels (mini-tools, custom
    // inspectors, plug-in UIs) without modifying Studio. Pass a stable
    // unique title, an externally-owned `bool* show` for menu-toggle, a
    // draw callback that runs every frame the window is shown, and a
    // dock-hint that the dockspace builder consults the first time the
    // layout is created.
    //
    // The studio takes ownership of nothing — `show` and the captured
    // state inside `draw` belong to the caller, who guarantees they
    // outlive the studio instance (or unregisters first).
    enum class ToolDockHint : u32 {
        Left = 0,        // dock into the left column with Hierarchy
        Right,           // dock into the right column with Inspector
        Bottom,          // dock into the bottom strip with Log / Console
        Center,          // dock into the central area with Viewports
        Floating         // don't dock — appear as its own free window
    };
    using ToolDrawFn = cardinal::function<void(bool* show)>;
    struct ToolWindow {
        cardinal::string  title;
        bool*        show{nullptr};
        ToolDrawFn   draw;
        ToolDockHint dock_hint{ToolDockHint::Right};
    };
    virtual void register_tool_window(ToolWindow tw) = 0;
    virtual void unregister_tool_window(const char* title) = 0;
    // Call inside the host's per-frame UI block (between begin_frame and
    // end_frame). Iterates registered tools, runs each draw callback when
    // *show is true.
    virtual void draw_tool_windows() = 0;
    // Call inside the host's View menu. Emits one ImGui::MenuItem per
    // registered tool window so users can show/hide them.
    virtual void draw_tool_windows_menu() = 0;
    // Read-only enumeration — used by the host's dock-layout builder so
    // it knows where to place each registered tool the first time the
    // layout is constructed.
    virtual cardinal::vector<ToolWindow> registered_tool_windows() const = 0;

    // ----- Diagnostics: stack / function trace / script debugger -------
    //
    // draw_stack_tracer_panel() — capture-on-demand call stack of the main
    //   thread (or any registered worker), symbolicated via DbgHelp on
    //   Windows. Useful when a plugin crashes (host SEH handler captures
    //   the stack and surfaces it here).
    //
    // draw_function_tracer_panel() — flame-style timeline of function
    //   call/return events from the cardinal::trace ring buffer. Both
    //   the Lua hook and any C++ scope wrapped in CARDINAL_TRACE_SCOPE
    //   feed this view.
    //
    // draw_script_debugger_panel(engine) — Lua JIT debugger UI: enable /
    //   disable, manage breakpoints, step in/out/over, inspect the paused
    //   stack + locals + upvalues.
    virtual void draw_stack_tracer_panel   (const char* title = "Stack Tracer",
                                            bool* p_open = nullptr) = 0;
    virtual void draw_function_tracer_panel(const char* title = "Function Tracer",
                                            bool* p_open = nullptr) = 0;
    virtual void draw_script_debugger_panel(script::Engine& engine,
                                            const char* title = "Script Debugger",
                                            bool* p_open = nullptr) = 0;

    // Runtime pipeline picker + per-pipeline knob editor + GPU feature
    // availability matrix. The registry owns the pipelines; the panel
    // mutates their Knob list directly so changes apply on the next
    // render() call. The bottom of the panel lists every cardinal::render
    // Feature with a green/grey badge based on the live GpuCapabilities.
    virtual void draw_render_pipeline_panel(render::Registry& reg,
                                            const char* title = "Render Pipeline",
                                            bool* p_open = nullptr) = 0;

protected:
    Studio() = default;
};

}  // namespace cardinal::ui
