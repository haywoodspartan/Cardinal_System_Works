// =============================================================================
// Cardinal — Sample 11: codebase stress world.
//
// A deliberately heavy StudioApplication that drives a broad slice of the
// runtime under load every frame so regressions in throughput / lifetime /
// integration show up immediately:
//
//   actor + sim + game : hundreds of GameActors (Spinner + RigidBody) ticked
//                         through SimWorld's tick groups + euler physics
//   player control     : the reusable actor::PlayerControllerComponent on a
//                         possessed PlayerActor (scripted patrol input here so
//                         it runs headless under the StudioEngine camera)
//   particles          : a high-rate emitter ticked each frame
//   virtual texturing  : a 4096-tile VT streamed with a wide per-frame request
//                         spread (ProceduralDecoder, no I/O)
//   net                : the SP=MMO loopback path — every actor's transform
//                         RepState broadcast + ingested per frame
//   navmesh            : a grid navmesh rebuilt + A* path-queried periodically
//
// Built on cardinal::ui::StudioEngine so window/device/swapchain/scene/
// pipelines come for free; this file is just the stress harness + a live
// dashboard panel. AssetCatalog teardown is handled by ~EngineImpl.
// =============================================================================
#include <cardinal/core/crash.hpp>
#include <cardinal/core/log.hpp>
#include <cardinal/engine/engine.hpp>
#include <cardinal/ui/studio_engine.hpp>

#include <cardinal/actor/world.hpp>
#include <cardinal/actor/component.hpp>
#include <cardinal/game/game.hpp>
#include <cardinal/game/game_actor.hpp>
#include <cardinal/game/reflection.hpp>
#include <cardinal/sim/sim.hpp>
#include <cardinal/particles/particles.hpp>
#include <cardinal/vt/vt.hpp>
#include <cardinal/net/net.hpp>
#include <cardinal/net/replication.hpp>
#include <cardinal/navmesh/navmesh.hpp>
#include <cardinal/scene/math.hpp>

#include <imgui.h>

namespace {

namespace clog = cardinal::log;

// StressSpinner — a RigidBody-bearing GameActor that also spins. Ticking
// hundreds of these exercises the actor component walk + SimWorld physics
// integrator + the Game lifecycle gate every frame.
class StressSpinner final : public cardinal::game::GameActor {
public:
    float rpm{45.0f};

    void on_tick(float dt) override {
        if (owner() == nullptr) return;
        if (auto* tr = owner()->get_component<
                cardinal::actor::TransformComponent>())
            tr->rotation_euler.y += rpm * 6.2831853f / 60.0f * dt;
    }
};
CARDINAL_REGISTER_GAME_CLASS(StressSpinner, "Stress/Spinner",
    PROP_FLOAT(rpm, 0.0f, 600.0f, "Revolutions per minute"))

// StressPlayer — owns the reusable PlayerControllerComponent. The harness
// feeds it a scripted patrol PlayerInput so the controller path is under
// load even though StudioEngine owns the real editor camera.
class StressPlayer final : public cardinal::game::GameActor {
public:
    void begin_play() override {
        if (owner() && owner()->get_component<
                cardinal::actor::PlayerControllerComponent>() == nullptr)
            owner()->add_component<
                cardinal::actor::PlayerControllerComponent>();
    }
};
CARDINAL_REGISTER_GAME_CLASS(StressPlayer, "Stress/Player", )

class StressApp final : public cardinal::ui::StudioApplication {
public:
    void on_init(cardinal::engine::Engine& /*e*/) override {
        rebuild_actors_();

        // Particles — one busy emitter.
        psys_ = cardinal::particles::System::create();
        {
            cardinal::particles::EmitterDesc d{};
            d.name            = "StressJet";
            d.origin          = {0.0f, 1.0f, 0.0f};
            d.rate_per_second = 4000.0f;
            d.max_particles   = 16384;
            d.lifetime_min = 0.5f; d.lifetime_max = 2.0f;
            d.size_min = 0.02f;    d.size_max = 0.08f;
            d.velocity_min = {-3.0f, 2.0f, -3.0f};
            d.velocity_max = { 3.0f, 8.0f,  3.0f};
            d.start_rgba = 0xFFFFC080u;
            d.end_rgba   = 0x0000A0FFu;
            psys_->add(d);
        }

        // Virtual texturing — a big VT streamed hard.
        {
            cardinal::vt::SystemDesc vd{};
            vd.pool_slots         = 4096;
            vd.prefetch_neighbors = 2;
            vts_ = cardinal::vt::System::create(vd);
            cardinal::vt::VirtualTextureDesc vtd{};
            vtd.id           = 0;
            vtd.width_tiles  = 4096;
            vtd.height_tiles = 4096;
            vtd.mip_count    = 13;
            vtd.decoder      = cardinal::make_shared<
                cardinal::vt::ProceduralDecoder>(0xC0FFEEu);
            vt_ = cardinal::vt::VirtualTexture::create(vtd);
            if (vts_ && vt_) vts_->register_vt(vt_);
        }

        // Net — the SP=MMO loopback path.
        net_ = cardinal::net::Transport::create_loopback();
        if (net_) {
            net_->listen(0);
            repl_ = cardinal::make_unique<cardinal::net::Replicator>(*net_);
        }

        build_navmesh_();
        game_.start_play();
        clog::infof("stress", "stress world online (%d actors)", actor_count_);
    }

    void on_simulate(cardinal::engine::Engine& /*e*/, cardinal::f32 dt) override {
        if (dt <= 0.0f) dt = 1.0f / 60.0f;
        sim_time_ += static_cast<double>(dt);

        // 1) actor/sim/game — spinners + RigidBody euler integration.
        game_.tick(dt);

        // 2) scripted player patrol — drives the reusable controller.
        if (auto* pa = sim_.world().find(player_id_)) {
            if (auto* pc = pa->get_component<
                    cardinal::actor::PlayerControllerComponent>()) {
                cardinal::actor::PlayerInput in{};
                in.accept_input = true;
                in.move_z = 1.0f;                  // walk forward
                patrol_t_ += dt;
                in.look     = true;
                in.mouse_dx = 90.0f * dt;          // gentle continuous turn
                if (patrol_t_ > 3.0f) { patrol_t_ = 0.0f; in.jump = true; }
                pc->tick(dt, in);
                player_pos_ = pc->camera_eye();
            }
        }

        // 3) particles.
        if (psys_) psys_->tick(dt);

        // 4) virtual texturing — request a wide spread, then pump.
        if (vts_ && vt_) {
            const cardinal::u32 mip = 3;
            const cardinal::u32 w = vt_->page_table().width_tiles(mip);
            const cardinal::u32 h = vt_->page_table().height_tiles(mip);
            const cardinal::u32 base = (vt_phase_++ * 7u);
            for (cardinal::u32 i = 0; i < 96u; ++i) {
                const cardinal::u32 tx = (base + i * 3u) % (w ? w : 1u);
                const cardinal::u32 ty = (base + i * 5u) % (h ? h : 1u);
                vt_->request(mip, ty, tx);
            }
            vts_->tick(static_cast<cardinal::u64>(sim_time_ * 1000.0));
        }

        // 5) net — replicate every actor's transform through loopback.
        if (repl_) {
            net_tx_.clear();
            for (const auto& aptr : sim_.world().actors()) {
                if (!aptr->alive()) continue;
                auto* tr = aptr->get_component<
                    cardinal::actor::TransformComponent>();
                if (tr == nullptr) continue;
                cardinal::net::RepState s;
                s.id             = aptr->id();
                s.position       = tr->translation;
                s.rotation_euler = tr->rotation_euler;
                s.scale          = tr->scale;
                net_tx_.push_back(s);
            }
            net_events_.clear();
            net_->poll(net_events_);
            net_rx_.clear();
            net_applied_ = static_cast<cardinal::u32>(
                repl_->client_ingest(net_events_, net_rx_));
            repl_->server_broadcast(net_tx_);
        }

        // 6) navmesh — periodic A* query across the grid.
        nav_accum_ += dt;
        if (nav_accum_ > 0.25f && !nm_.empty()) {
            nav_accum_ = 0.0f;
            nav_wp_.clear();
            cardinal::navmesh::PathQuery pq;
            const auto st = pq.find_path(nm_,
                {-18.0f, 0.0f, -18.0f}, {18.0f, 0.0f, 18.0f}, nav_wp_);
            nav_found_     = st.found;
            nav_waypoints_ = st.waypoint_count;
            nav_cost_      = st.poly_path_cost;
        }
    }

    void on_ui(cardinal::ui::StudioEngine& /*se*/) override {
        if (!show_panel_) return;
        if (ImGui::Begin("Codebase Stress", &show_panel_)) {
            ImGui::Text("sim time      : %.1f s", sim_time_);
            ImGui::Separator();
            ImGui::Text("game actors   : %u (state=%s)",
                game_.game_actor_count(),
                cardinal::game::game_state_name(game_.state()));
            ImGui::Text("player eye    : %.1f, %.1f, %.1f",
                player_pos_.x, player_pos_.y, player_pos_.z);
            if (psys_) {
                const auto ps = psys_->stats();
                ImGui::Text("particles     : %llu spawned total",
                    (unsigned long long)ps.particles_total_spawned);
            }
            if (vts_) {
                const auto vs = vts_->stats();
                ImGui::Text("vt tiles      : seen=%llu processed=%llu "
                            "resident=%u",
                    (unsigned long long)vs.requests_seen,
                    (unsigned long long)vs.requests_processed,
                    vs.cache.resident);
            }
            if (repl_) {
                ImGui::Text("net snapshots : sent=%llu recv=%llu "
                            "applied=%u  wire=%u B",
                    (unsigned long long)repl_->snapshots_sent(),
                    (unsigned long long)repl_->snapshots_recv(),
                    net_applied_, repl_->last_snapshot_bytes());
            }
            ImGui::Text("navmesh path  : %s, %u wp, cost %.1f",
                nav_found_ ? "found" : "none", nav_waypoints_, nav_cost_);
            ImGui::Separator();
            ImGui::SliderInt("Spinner actors", &want_actors_, 16, 4000);
            if (ImGui::Button("Rebuild world")) rebuild_actors_();
        }
        ImGui::End();
    }

    void on_menu_bar(cardinal::ui::StudioEngine& /*se*/) override {
        if (ImGui::BeginMenu("Stress")) {
            ImGui::MenuItem("Dashboard", nullptr, &show_panel_);
            ImGui::EndMenu();
        }
    }

private:
    void rebuild_actors_() {
        // Fresh SimWorld+Game so a rebuild can't double-spawn. (Declared
        // members are reset; Game re-binds to the new SimWorld.)
        game_.stop_play();
        // Re-seed: spawn want_actors_ spinners in a grid + one player.
        actor_count_ = 0;
        for (int i = 0; i < want_actors_; ++i) {
            auto* a = game_.spawn_class("StressSpinner", "Spinner");
            if (a == nullptr) break;
            if (auto* s = a->get_component<StressSpinner>())
                s->rpm = 20.0f + static_cast<float>(i % 90);
            if (auto* tr = a->get_component<
                    cardinal::actor::TransformComponent>()) {
                const int gx = i % 40, gz = i / 40;
                tr->translation = { (gx - 20) * 1.5f, 1.0f,
                                    (gz - 20) * 1.5f };
            }
            auto* rb = a->add_component<
                cardinal::actor::RigidBodyComponent>();
            rb->use_gravity = false;       // keep them in the grid; still
            rb->kinematic   = false;       // walked by the physics group
            auto* m = a->add_component<cardinal::actor::MeshComponent>();
            m->asset_id = "primitives/box";
            ++actor_count_;
        }
        if (auto* p = game_.spawn_class("StressPlayer", "Player")) {
            player_id_ = p->id();
            if (auto* tr = p->get_component<
                    cardinal::actor::TransformComponent>())
                tr->translation = {0.0f, 0.0f, 0.0f};
        }
        if (game_.state() != cardinal::game::GameState::Playing)
            game_.start_play();
        clog::infof("stress", "rebuilt: %d spinners", actor_count_);
    }

    void build_navmesh_() {
        // A flat 40x40 quad grid (two tris per cell) → navmesh.
        cardinal::vector<cardinal::scene::Vec3> verts;
        cardinal::vector<cardinal::u32>         idx;
        constexpr int N = 20;
        for (int z = 0; z <= N; ++z)
            for (int x = 0; x <= N; ++x)
                verts.push_back({ (x - N / 2) * 2.0f, 0.0f,
                                  (z - N / 2) * 2.0f });
        auto vid = [&](int x, int z) {
            return static_cast<cardinal::u32>(z * (N + 1) + x);
        };
        for (int z = 0; z < N; ++z)
            for (int x = 0; x < N; ++x) {
                idx.push_back(vid(x,   z));
                idx.push_back(vid(x+1, z));
                idx.push_back(vid(x+1, z+1));
                idx.push_back(vid(x,   z));
                idx.push_back(vid(x+1, z+1));
                idx.push_back(vid(x,   z+1));
            }
        nm_.build_from_triangles(verts, idx);
        clog::infof("stress", "navmesh: %llu polys",
            (unsigned long long)nm_.polys.size());
    }

    // sim_ must outlive game_ (Game holds a SimWorld&).
    cardinal::sim::SimWorld sim_{};
    cardinal::game::Game    game_{sim_};

    cardinal::shared_ptr<cardinal::particles::System>     psys_;
    cardinal::shared_ptr<cardinal::vt::System>            vts_;
    cardinal::shared_ptr<cardinal::vt::VirtualTexture>    vt_;
    cardinal::unique_ptr<cardinal::net::Transport>        net_;
    cardinal::unique_ptr<cardinal::net::Replicator>       repl_;
    cardinal::vector<cardinal::net::NetEvent>             net_events_;
    cardinal::vector<cardinal::net::RepState>             net_tx_, net_rx_;
    cardinal::navmesh::Mesh                               nm_;
    cardinal::vector<cardinal::scene::Vec3>               nav_wp_;

    cardinal::actor::ActorId player_id_{0};
    cardinal::scene::Vec3    player_pos_{0, 0, 0};
    int      want_actors_{512};
    int      actor_count_{0};
    double   sim_time_{0.0};
    float    patrol_t_{0.0f};
    float    nav_accum_{0.0f};
    cardinal::u32 vt_phase_{0};
    cardinal::u32 net_applied_{0};
    bool     nav_found_{false};
    cardinal::u32 nav_waypoints_{0};
    float    nav_cost_{0.0f};
    bool     show_panel_{true};
};

}  // namespace

int main(int argc, char** argv) {
    cardinal::crash::install();
    StressApp app;
    return cardinal::ui::StudioEngine::run(argc, argv, app);
}
