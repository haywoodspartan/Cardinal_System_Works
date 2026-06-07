// =============================================================================
// 03_studio sample — engine console registration.
//
// Lifted out of main.cpp so the sample's frame loop stays readable. Every
// CARDINAL_CVAR_*/CARDINAL_CCOMMAND macro in this file binds a name in the
// process-wide cardinal::console::Registry to a getter / setter / functor
// that captures the live engine state via the ConsoleSetupContext.
//
// The macros lower to lambdas that capture by value — so the context is
// snapshotted into each registration; once `register_engine_console_commands`
// returns the ctx struct itself can go out of scope (its pointers must stay
// valid, but the struct doesn't).
// =============================================================================
#include "sample_types.hpp"

#include <cardinal/cmd/command.hpp>
#include <cardinal/console/console.hpp>
#include <cardinal/core/frame_pacer.hpp>
#include <cardinal/cppscript/cppscript.hpp>
#include <cardinal/plugin/plugin.hpp>
#include <cardinal/render/pipeline.hpp>
#include <cardinal/rhi/rhi.hpp>
#include <cardinal/scene/assets.hpp>
#include <cardinal/scene/scene.hpp>
#include <cardinal/ui/studio.hpp>
#include <cardinal/world/world.hpp>

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace sample_studio {

namespace cv  = cardinal::console;
namespace rhi = cardinal::rhi;
namespace rnd = cardinal::render;
namespace scn = cardinal::scene;

using cardinal::u32;
using cardinal::i32;
using cardinal::i64;
using cardinal::f64;
using cardinal::usize;

// Find a knob on the AEGIS pipeline by id (nullptr when AEGIS isn't the
// registered pipeline or the knob is absent) — bridges AEGIS knobs to
// r.aegis.* cvars so they're settable from the console + Options window +
// persisted via config.save. The bridge's apply_config() reads the knob each
// frame, so a cvar set propagates into the live AegisConfig.
static rnd::Knob* aegis_knob(rnd::Registry* pipelines, const char* id) {
    if (!pipelines) return nullptr;
    for (auto* p : pipelines->all())
        if (p && p->id() == rnd::PipelineId::Aegis)
            for (auto& k : p->knobs())
                if (k.id == id) return &k;
    return nullptr;
}

void register_engine_console_commands(const ConsoleSetupContext& ctx) {
    // Snapshot pointers locally so each lambda captures by value and stays
    // independent of the ctx parameter's lifetime.
    auto* sw                   = ctx.sw;
    auto* pacer                = ctx.pacer;
    auto* pipelines            = ctx.pipelines;
    auto* world_grid           = ctx.world_grid;
    auto* world_streamer       = ctx.world_streamer;
    auto* scene                = ctx.scene;
    auto* device               = ctx.device;
    auto* viewports            = ctx.viewports;
    auto* next_viewport_serial = ctx.next_viewport_serial;
    auto* selected_id          = ctx.selected_id;
    auto* want_quit            = ctx.want_quit;
    auto* studio               = ctx.studio;
    auto* cppscript            = ctx.cppscript;
    auto* commands             = ctx.commands;
    auto* aworld               = ctx.aworld;
    auto* placement            = ctx.placement;
    auto* undo                 = ctx.undo;
    auto* selected_actor_id    = ctx.selected_actor_id;

    // Helper used by viewport.add / viewport.count to materialise new slots
    // without duplicating the title formatting.
    auto make_slot = [next_viewport_serial](scn::ViewMode m) -> ViewportSlot {
        ViewportSlot s;
        s.title = "Viewport " + std::to_string((*next_viewport_serial)++);
        s.mode  = m;
        return s;
    };

    // ---- Renderer / swapchain ---------------------------------------------
    CARDINAL_CVAR_BOOL("r.vsync",
        "Vertical sync (1 = locked to display, 0 = uncapped + tearing)",
        [sw] { return sw->vsync(); },
        [sw](bool v) { sw->set_vsync(v); });
    CARDINAL_CVAR_INT("r.vsync_interval",
        "VSync interval — 0=off, 1=locked, 2=half, 3=third, 4=quarter",
        0, 4,
        [sw] { return static_cast<i64>(sw->vsync_interval()); },
        [sw](i64 v) { sw->set_vsync_interval(static_cast<u32>(v)); });
    if (sw->reflex_supported()) {
        CARDINAL_CVAR_INT("r.reflex_mode",
            "NVIDIA Reflex — 0=Off, 1=On, 2=On+Boost",
            0, 2,
            [sw] { return static_cast<i64>(sw->reflex_mode()); },
            [sw](i64 v) { sw->set_reflex_mode(static_cast<rhi::Swapchain::ReflexMode>(v)); });
        CARDINAL_CVAR_INT("r.reflex_fps_cap",
            "Reflex FPS cap (0 = disabled)",
            0, 480,
            [sw] { return static_cast<i64>(sw->reflex_fps_cap()); },
            [sw](i64 v) { sw->set_reflex_fps_cap(static_cast<u32>(v)); });
    }
    CARDINAL_CVAR_FLOAT("r.fps_cap_main_bg",
        "Main loop background FPS cap (when window unfocused)",
        0.0, 480.0,
        [pacer] {
            return static_cast<f64>(pacer->limit(cardinal::core::kPaceMainLoop).fps_background);
        },
        [pacer](f64 v) {
            auto lim = pacer->limit(cardinal::core::kPaceMainLoop);
            lim.fps_background = static_cast<float>(v);
            pacer->set_limit(cardinal::core::kPaceMainLoop, lim);
        });
    CARDINAL_CVAR_FLOAT("r.fps_cap_main_fg",
        "Main loop foreground FPS cap (0 = unlimited)",
        0.0, 480.0,
        [pacer] {
            return static_cast<f64>(pacer->limit(cardinal::core::kPaceMainLoop).fps_foreground);
        },
        [pacer](f64 v) {
            auto lim = pacer->limit(cardinal::core::kPaceMainLoop);
            lim.fps_foreground = static_cast<float>(v);
            pacer->set_limit(cardinal::core::kPaceMainLoop, lim);
        });

    // ---- Pipeline ---------------------------------------------------------
    CARDINAL_CVAR_INT("pipeline.active",
        "Active render pipeline id (see `pipeline.list`)",
        0, 32,
        [pipelines] { return static_cast<i64>(pipelines->active_id()); },
        [pipelines](i64 v) { pipelines->set_active(static_cast<rnd::PipelineId>(v)); });
    CARDINAL_CVAR_INT("pipeline.view_mode",
        "Active pipeline's view mode — 0=Solid 1=Wireframe 2=Polygons "
        "3=Heightmap 4=Normals 5=RTXPreview",
        0, 5,
        [pipelines]() -> i64 {
            if (auto* p = pipelines->active()) {
                for (auto& k : p->knobs())
                    if (k.id == "view_mode") return static_cast<i64>(k.e);
            }
            return 0;
        },
        [pipelines](i64 v) {
            if (auto* p = pipelines->active()) {
                for (auto& k : p->knobs())
                    if (k.id == "view_mode") { k.e = static_cast<int>(v); return; }
            }
        });
    CARDINAL_CCOMMAND("pipeline.list",
        "List every registered render pipeline + its id",
        [pipelines](const std::vector<std::string>&, cv::Output& out) {
            for (auto* p : pipelines->all()) {
                out("%u  %s — %s",
                    static_cast<unsigned>(p->id()), p->name(), p->description());
            }
        });

    // ---- AEGIS Pipeline 2.0 GPU features + precision tier (cvars) ----------
    // Surface the AEGIS knobs as r.aegis.* cvars so the GPU features
    // (DirectStorage / Bindless / Async / VRS / FP8 / FP4) + quality knobs are
    // settable from the console AND the Options/Settings window AND persisted
    // via config.save. Each bridges to the live AEGIS knob; the pipeline's
    // apply_config() then propagates the change into the AegisConfig each frame
    // (so e.g. `set r.aegis.geometry_tier 3` escalates the tier to FP4).
    CARDINAL_CVAR_BOOL("r.aegis.async_compute",
        "AEGIS async-compute feature (device-gated)",
        [pipelines]{ auto* k = aegis_knob(pipelines, "async_compute"); return k && k->b; },
        [pipelines](bool v){ if (auto* k = aegis_knob(pipelines, "async_compute")) k->b = v; });
    CARDINAL_CVAR_BOOL("r.aegis.variable_rate_shading",
        "AEGIS variable-rate shading (device-gated)",
        [pipelines]{ auto* k = aegis_knob(pipelines, "variable_rate_shading"); return k && k->b; },
        [pipelines](bool v){ if (auto* k = aegis_knob(pipelines, "variable_rate_shading")) k->b = v; });
    CARDINAL_CVAR_BOOL("r.aegis.bindless_resources",
        "AEGIS bindless resources (device-gated)",
        [pipelines]{ auto* k = aegis_knob(pipelines, "bindless_resources"); return k && k->b; },
        [pipelines](bool v){ if (auto* k = aegis_knob(pipelines, "bindless_resources")) k->b = v; });
    CARDINAL_CVAR_BOOL("r.aegis.direct_storage",
        "AEGIS DirectStorage / RTX IO streaming (device-gated)",
        [pipelines]{ auto* k = aegis_knob(pipelines, "direct_storage"); return k && k->b; },
        [pipelines](bool v){ if (auto* k = aegis_knob(pipelines, "direct_storage")) k->b = v; });
    CARDINAL_CVAR_BOOL("r.aegis.allow_fp8",
        "AEGIS: permit the geometry tier to escalate to FP8 (device-gated)",
        [pipelines]{ auto* k = aegis_knob(pipelines, "max_tier_fp8"); return k && k->b; },
        [pipelines](bool v){ if (auto* k = aegis_knob(pipelines, "max_tier_fp8")) k->b = v; });
    CARDINAL_CVAR_BOOL("r.aegis.allow_fp4",
        "AEGIS: permit the geometry tier to escalate to FP4 (device-gated)",
        [pipelines]{ auto* k = aegis_knob(pipelines, "max_tier_fp4"); return k && k->b; },
        [pipelines](bool v){ if (auto* k = aegis_knob(pipelines, "max_tier_fp4")) k->b = v; });
    CARDINAL_CVAR_INT("r.aegis.geometry_tier",
        "AEGIS max geometry tier — 0=FP32 1=FP16 2=FP8 3=FP4 (clamped to device)",
        0, 3,
        [pipelines]() -> i64 { auto* k = aegis_knob(pipelines, "max_tier"); return k ? static_cast<i64>(k->e) : 1; },
        [pipelines](i64 v){ if (auto* k = aegis_knob(pipelines, "max_tier")) k->e = static_cast<int>(v); });
    CARDINAL_CVAR_FLOAT("r.aegis.exposure",
        "AEGIS tonemap exposure (EV)",
        0.0, 4.0,
        [pipelines]() -> f64 { auto* k = aegis_knob(pipelines, "exposure"); return k ? static_cast<f64>(k->f) : 1.0; },
        [pipelines](f64 v){ if (auto* k = aegis_knob(pipelines, "exposure")) k->f = static_cast<float>(v); });
    CARDINAL_CVAR_FLOAT("r.aegis.resolution_scale",
        "AEGIS internal resolution scale (0.5 = quarter pixels, 2.0 = SSAA x2)",
        0.5, 2.0,
        [pipelines]() -> f64 { auto* k = aegis_knob(pipelines, "resolution_scale"); return k ? static_cast<f64>(k->f) : 1.0; },
        [pipelines](f64 v){ if (auto* k = aegis_knob(pipelines, "resolution_scale")) k->f = static_cast<float>(v); });

    // ---- World streaming --------------------------------------------------
    CARDINAL_CVAR_FLOAT("world.chunk_size",
        "World grid chunk size (units)",
        1.0, 4096.0,
        [world_grid] { return static_cast<f64>(world_grid->chunk_size()); },
        [world_grid, world_streamer](f64 v) {
            world_grid->set_chunk_size(static_cast<float>(v));
            world_streamer->invalidate();
        });
    CARDINAL_CVAR_INT("world.render_dist_xz",
        "World streaming horizontal radius (chunks)",
        0, 64,
        [world_grid] { return static_cast<i64>(world_grid->render_distance_xz()); },
        [world_grid, world_streamer](i64 v) {
            world_grid->set_render_distance_xz(static_cast<i32>(v));
            world_streamer->invalidate();
        });
    CARDINAL_CVAR_INT("world.render_dist_y",
        "World streaming vertical half-height (chunks)",
        0, 32,
        [world_grid] { return static_cast<i64>(world_grid->render_distance_y()); },
        [world_grid, world_streamer](i64 v) {
            world_grid->set_render_distance_y(static_cast<i32>(v));
            world_streamer->invalidate();
        });
    CARDINAL_CVAR_INT("world.extent_x",
        "World grid half-extent on X (chunks)",
        1, 1<<29,
        [world_grid] { return static_cast<i64>(world_grid->extent_x()); },
        [world_grid](i64 v) { world_grid->set_extent_x(static_cast<i32>(v)); });
    CARDINAL_CVAR_INT("world.extent_y",
        "World grid half-extent on Y (chunks)",
        1, 1<<29,
        [world_grid] { return static_cast<i64>(world_grid->extent_y()); },
        [world_grid](i64 v) { world_grid->set_extent_y(static_cast<i32>(v)); });
    CARDINAL_CVAR_INT("world.extent_z",
        "World grid half-extent on Z (chunks)",
        1, 1<<29,
        [world_grid] { return static_cast<i64>(world_grid->extent_z()); },
        [world_grid](i64 v) { world_grid->set_extent_z(static_cast<i32>(v)); });
    CARDINAL_CCOMMAND("world.evict_all",
        "Force-unload every active streamed chunk",
        [world_streamer](const std::vector<std::string>&, cv::Output& out) {
            world_streamer->evict_all();
            out("evicted; active=%zu", world_streamer->active_count());
        });
    // ---- Command bus bridge — dispatch any registered editor command -----
    // `cmd list` enumerates; `cmd <id>` dispatches with a context mirroring
    // the Ctrl+P palette (actor world + scene + placement + undo + current
    // selection). Makes the whole framework scriptable from the console.
    CARDINAL_CCOMMAND("cmd",
        "cmd <id> — dispatch an editor command (or `cmd list` to enumerate)",
        [commands, aworld, scene, placement, undo, selected_id, selected_actor_id]
        (const std::vector<std::string>& argv, cv::Output& out) {
            if (commands == nullptr) { out("cmd: no command registry bound"); return; }
            if (argv.size() < 2 || argv[1] == "list") {
                for (const auto& id : commands->ids()) {
                    const auto* c = commands->find(id);
                    out("  %-26s %s", id.c_str(),
                        (c && !c->label.empty()) ? c->label.c_str() : "");
                }
                return;
            }
            const cardinal::string id = argv[1].c_str();
            cardinal::cmd::CommandContext cx{};
            cx.world      = aworld;
            cx.scene      = scene;
            cx.placement  = placement;
            cx.scene_undo = undo;
            if (selected_actor_id && *selected_actor_id != 0)
                cx.selection = { *selected_actor_id };
            if (selected_id && *selected_id != 0)
                cx.scene_selection = { *selected_id };
            const auto r = commands->dispatch(id, cx);
            out("cmd %s -> %s%s%s  [n=%u]", id.c_str(), r.ok ? "ok" : "FAIL",
                r.message.empty() ? "" : ": ",
                r.message.empty() ? "" : r.message.c_str(), cx.result_count);
        });
    // ---- Settings persistence (UE5-style config) -------------------------
    // The default settings file the Studio auto-loads at startup + auto-saves
    // on exit (see main.cpp). config.save/load take an optional explicit path.
    CARDINAL_CCOMMAND("config.save",
        "config.save [file] — write all cvars to a settings file (default studio_settings.cfg)",
        [](const std::vector<std::string>& argv, cv::Output& out) {
            const cardinal::string path =
                (argv.size() > 1) ? cardinal::string(argv[1].c_str())
                                  : cardinal::string("studio_settings.cfg");
            const auto n = cv::Registry::instance().save_cvars(path);
            out("config.save: wrote %zu cvars to %s",
                static_cast<usize>(n), path.c_str());
        });
    CARDINAL_CCOMMAND("config.load",
        "config.load [file] — apply a settings file (default studio_settings.cfg)",
        [](const std::vector<std::string>& argv, cv::Output& out) {
            const cardinal::string path =
                (argv.size() > 1) ? cardinal::string(argv[1].c_str())
                                  : cardinal::string("studio_settings.cfg");
            const auto n = cv::Registry::instance().exec_file(path, out);
            out("config.load: applied %zu lines from %s",
                static_cast<usize>(n), path.c_str());
        });
    CARDINAL_CCOMMAND("exec",
        "exec <file> — run a console script: replay each line of the file",
        [](const std::vector<std::string>& argv, cv::Output& out) {
            if (argv.size() < 2) { out("usage: exec <file>"); return; }
            const cardinal::string path(argv[1].c_str());
            const auto n = cv::Registry::instance().exec_file(path, out);
            out("exec %s: %zu lines ran", path.c_str(), static_cast<usize>(n));
        });
    CARDINAL_CCOMMAND("world.stats",
        "Print world-streaming stats",
        [world_grid, world_streamer](const std::vector<std::string>&, cv::Output& out) {
            const auto cc = world_streamer->camera_chunk();
            out("camera_chunk = (%d, %d, %d)", cc.x, cc.y, cc.z);
            out("active = %zu / %lld total",
                world_streamer->active_count(),
                static_cast<long long>(world_grid->total_chunk_count()));
            out("world_size = %.0f x %.0f x %.0f units",
                world_grid->world_size_x_units(),
                world_grid->world_size_y_units(),
                world_grid->world_size_z_units());
        });

    // ---- Viewports --------------------------------------------------------
    CARDINAL_CVAR_INT("viewport.count",
        "Number of editor viewport panels (1..16)",
        1, kMaxViewports,
        [viewports] { return static_cast<i64>(viewports->size()); },
        [viewports, make_slot](i64 v) {
            const int target = static_cast<int>(std::clamp<i64>(v, 1, kMaxViewports));
            while (static_cast<int>(viewports->size()) < target)
                viewports->push_back(make_slot(scn::ViewMode::Solid));
            while (static_cast<int>(viewports->size()) > target)
                viewports->pop_back();
        });
    CARDINAL_CCOMMAND("viewport.add",
        "Append a new viewport slot (up to the configured max)",
        [viewports, make_slot](const std::vector<std::string>&, cv::Output& out) {
            if (static_cast<int>(viewports->size()) >= kMaxViewports) {
                out("error: at max (%d) viewports", kMaxViewports);
                return;
            }
            viewports->push_back(make_slot(scn::ViewMode::Solid));
            out("ok; count=%zu", viewports->size());
        });
    CARDINAL_CCOMMAND("viewport.remove",
        "Remove the trailing viewport slot",
        [viewports](const std::vector<std::string>&, cv::Output& out) {
            if (viewports->size() <= 1) {
                out("error: at least one viewport must remain");
                return;
            }
            viewports->pop_back();
            out("ok; count=%zu", viewports->size());
        });
    CARDINAL_CCOMMAND("viewport.maximize",
        "Toggle maximize on viewport <id>; with no arg, prints current state",
        [viewports](const std::vector<std::string>& argv, cv::Output& out) {
            if (argv.size() == 1) {
                for (usize i = 0; i < viewports->size(); ++i) {
                    out("[%zu] %s — %s%s%s", i, (*viewports)[i].title.c_str(),
                        (*viewports)[i].show ? "shown" : "hidden",
                        (*viewports)[i].maximized ? " maximized" : "",
                        (*viewports)[i].minimized ? " minimized" : "");
                }
                return;
            }
            i64 id = -1;
            if (argv.size() >= 2) {
                char* end = nullptr;
                id = std::strtoll(argv[1].c_str(), &end, 0);
            }
            if (id < 0 || id >= static_cast<i64>(viewports->size())) {
                out("error: id out of range (0..%zu)", viewports->size() - 1);
                return;
            }
            // Only one viewport can be maximized at a time — if we're
            // about to maximize this one, demote any siblings first.
            if (!(*viewports)[id].maximized) {
                for (auto& vps : *viewports) vps.maximized = false;
            }
            (*viewports)[id].maximized = !(*viewports)[id].maximized;
            out("viewport %lld: %smaximized",
                static_cast<long long>(id),
                (*viewports)[id].maximized ? "" : "un-");
        });

    // ---- Scene queries ----------------------------------------------------
    CARDINAL_CCOMMAND("scene.list",
        "List every entity (id, name, parent, visibility, position)",
        [scene](const std::vector<std::string>&, cv::Output& out) {
            out("%zu entities", scene->entities().size());
            for (const auto& e : scene->entities()) {
                out("%-5u  %-24s  parent=%-5u  %svis  pos=(%.2f, %.2f, %.2f)",
                    e.id, e.name.c_str(), e.parent_id,
                    e.visible ? " " : "in",
                    e.transform.translation.x,
                    e.transform.translation.y,
                    e.transform.translation.z);
            }
        });
    CARDINAL_CCOMMAND("scene.delete",
        "Delete entity by id: scene.delete <id>",
        [scene](const std::vector<std::string>& argv, cv::Output& out) {
            if (argv.size() < 2) { out("usage: scene.delete <id>"); return; }
            u32 id = static_cast<u32>(std::strtoul(argv[1].c_str(), nullptr, 0));
            if (scene->remove_entity(id)) out("removed entity %u", id);
            else                          out("error: no entity with id %u", id);
        });
    CARDINAL_CCOMMAND("scene.select",
        "Select entity by id: scene.select <id>",
        [scene, selected_id](const std::vector<std::string>& argv, cv::Output& out) {
            if (argv.size() < 2) { out("usage: scene.select <id>"); return; }
            u32 id = static_cast<u32>(std::strtoul(argv[1].c_str(), nullptr, 0));
            if (id == 0 || scene->find_by_id(id) != nullptr) {
                *selected_id = id;
                out("selected = %u", id);
            } else {
                out("error: no entity with id %u", id);
            }
        });
    // Note: the lambda body avoids brace-initialisers — {x, y, z} would
    // confuse the preprocessor's macro tokenisation (it splits on commas
    // inside the FN argument). Per-axis writes side-step that.
    CARDINAL_CCOMMAND("scene.spawn",
        "Spawn an asset from the catalog: scene.spawn <asset_id> [x y z]",
        [scene, device, selected_id](const std::vector<std::string>& argv, cv::Output& out) {
            if (argv.size() < 2) { out("usage: scene.spawn <asset_id> [x y z]"); return; }
            scn::AssetSpawnContext sctx{};
            sctx.scene = scene; sctx.device = device;
            if (argv.size() >= 5) {
                sctx.position.x = static_cast<float>(std::strtod(argv[2].c_str(), nullptr));
                sctx.position.y = static_cast<float>(std::strtod(argv[3].c_str(), nullptr));
                sctx.position.z = static_cast<float>(std::strtod(argv[4].c_str(), nullptr));
            }
            auto r = scn::AssetCatalog::instance().spawn(argv[1].c_str(), sctx);
            if (r.primary_id != 0) {
                *selected_id = r.primary_id;
                out("spawned %s (id=%u)", argv[1].c_str(), r.primary_id);
            } else {
                out("error: catalog has no entry '%s'", argv[1].c_str());
            }
        });

    // ---- Editor + lifecycle ----------------------------------------------
    CARDINAL_CCOMMAND("editor.quit",
        "Close the editor window (graceful shutdown)",
        [want_quit](const std::vector<std::string>&, cv::Output& out) {
            *want_quit = true;
            out("shutting down…");
        });
    // ---- Plugin runtime --------------------------------------------------
    CARDINAL_CCOMMAND("plugin.list",
        "List every loaded plugin: name, version, path, status.",
        [](const std::vector<std::string>&, cv::Output& out) {
            const auto entries = cardinal::plugin::Registry::instance().enumerate();
            if (entries.empty()) { out("(no plugins loaded)"); return; }
            for (const auto& e : entries) {
                out("%-24s %-8s %s%s",
                    e.name.empty() ? "(unnamed)" : e.name.c_str(),
                    e.version.empty() ? "?" : e.version.c_str(),
                    e.path.c_str(),
                    e.disabled ? "  [DISABLED — tick crashed]" : "");
            }
        });
    CARDINAL_CCOMMAND("plugin.unload",
        "Unload a plugin by name or path: plugin.unload <name-or-path>",
        [](const std::vector<std::string>& argv, cv::Output& out) {
            if (argv.size() < 2) { out("usage: plugin.unload <name-or-path>"); return; }
            const bool ok = cardinal::plugin::Registry::instance().unload(argv[1].c_str());
            out("%s", ok ? "ok" : "error: not loaded");
        });
    CARDINAL_CCOMMAND("plugin.reload",
        "Reload a plugin (unload + load): plugin.reload <name-or-path>",
        [](const std::vector<std::string>& argv, cv::Output& out) {
            if (argv.size() < 2) { out("usage: plugin.reload <name-or-path>"); return; }
            const bool ok = cardinal::plugin::Registry::instance().reload(argv[1].c_str());
            out("%s", ok ? "ok" : "error: reload failed");
        });

    // ---- C++ scripting (cardinal::cppscript) -----------------------------
    CARDINAL_CCOMMAND("script.compiler",
        "Print the C++ compiler the script engine resolved to.",
        [cppscript](const std::vector<std::string>&, cv::Output& out) {
            if (cppscript == nullptr) { out("error: cppscript not bound"); return; }
            const std::string p = cppscript->compiler_path();
            if (p.empty()) {
                out("(no compiler — set CARDINAL_CPPSCRIPT_COMPILER or install VS 2022 / clang-cl)");
            } else {
                out("compiler: %s", p.c_str());
            }
        });
    CARDINAL_CCOMMAND("script.compile",
        "Compile + load a C++ script: script.compile <path-to-.cpp>",
        [cppscript](const std::vector<std::string>& argv, cv::Output& out) {
            if (cppscript == nullptr) { out("error: cppscript not bound"); return; }
            if (argv.size() < 2) { out("usage: script.compile <path-to-.cpp>"); return; }
            auto h = cppscript->compile_and_load(argv[1].c_str());
            if (!h) { out("error: failed to enqueue compile job"); return; }
            out("queued job %llu — poll with `script.list`",
                static_cast<unsigned long long>(h.id));
        });
    CARDINAL_CCOMMAND("script.unload",
        "Unload a script by source path or by plugin name: script.unload <name-or-path>",
        [cppscript](const std::vector<std::string>& argv, cv::Output& out) {
            if (cppscript == nullptr) { out("error: cppscript not bound"); return; }
            if (argv.size() < 2) { out("usage: script.unload <name-or-path>"); return; }
            const bool ok = cppscript->unload(argv[1].c_str());
            out("%s", ok ? "ok" : "error: not loaded");
        });
    CARDINAL_CCOMMAND("script.list",
        "List every cppscript job: status, exit code, source.",
        [cppscript](const std::vector<std::string>&, cv::Output& out) {
            if (cppscript == nullptr) { out("error: cppscript not bound"); return; }
            const auto jobs = cppscript->jobs();
            if (jobs.empty()) { out("(no scripts queued)"); return; }
            auto status_name = [](cardinal::cppscript::JobStatus s) {
                using S = cardinal::cppscript::JobStatus;
                switch (s) {
                    case S::Pending:        return "Pending";
                    case S::Compiling:      return "Compiling";
                    case S::LoadPending:    return "LoadPending";
                    case S::Loaded:         return "Loaded";
                    case S::CompileFailed:  return "CompileFailed";
                    case S::LoadFailed:     return "LoadFailed";
                }
                return "?";
            };
            for (const auto& j : jobs) {
                out("[%llu] %-14s rc=%-4lld %s",
                    static_cast<unsigned long long>(j.handle.id),
                    status_name(j.status),
                    static_cast<long long>(j.exit_code),
                    j.source_path.c_str());
                if (j.status == cardinal::cppscript::JobStatus::CompileFailed &&
                    !j.diagnostics.empty()) {
                    // Print first 4 lines of diagnostics so the user can see
                    // the error inline without flooding the console.
                    int lines = 0;
                    for (size_t i = 0, e = j.diagnostics.size(); i < e && lines < 4; ++i) {
                        size_t nl = j.diagnostics.find('\n', i);
                        if (nl == std::string::npos) nl = e;
                        out("    %s", j.diagnostics.substr(i, nl - i).c_str());
                        i = nl;
                        ++lines;
                    }
                    if (lines == 4) out("    ... (truncated)");
                }
            }
        });
    CARDINAL_CCOMMAND("script.watch",
        "Watch a .cpp file: changes trigger recompile + reload. "
        "script.watch <path>",
        [cppscript](const std::vector<std::string>& argv, cv::Output& out) {
            if (cppscript == nullptr) { out("error: cppscript not bound"); return; }
            if (argv.size() < 2) { out("usage: script.watch <path>"); return; }
            cppscript->watch(argv[1].c_str());
            out("watching %s", argv[1].c_str());
        });
    CARDINAL_CCOMMAND("script.unwatch",
        "Stop watching a path: script.unwatch <path>",
        [cppscript](const std::vector<std::string>& argv, cv::Output& out) {
            if (cppscript == nullptr) { out("error: cppscript not bound"); return; }
            if (argv.size() < 2) { out("usage: script.unwatch <path>"); return; }
            cppscript->unwatch(argv[1].c_str());
            out("unwatched %s", argv[1].c_str());
        });

    CARDINAL_CCOMMAND("editor.reset_layout",
        "Wipe Studio's persisted window layout (positions, dock state). "
        "Useful when a panel is stuck off-screen or the .ini was corrupted.",
        [studio](const std::vector<std::string>&, cv::Output& out) {
            if (studio == nullptr) { out("error: no studio bound"); return; }
            studio->reset_layout();
            out("layout reset — windows will reposition on the next frame.");
        });
    CARDINAL_CCOMMAND("editor.set_backend",
        "Switch render backend at runtime: editor.set_backend vulkan|d3d12.",
        [sw](const std::vector<std::string>& argv, cv::Output& out) {
            if (argv.size() < 2) {
                out("usage: editor.set_backend <vulkan|d3d12|toggle>");
                out("current: %s", sw == nullptr ? "(none)" :
                    "Vulkan");   // Live reading would need a Device& not a Swapchain*
                return;
            }
            // Backend hot-swap is fully wired up at the engine level
            // (cardinal::engine::Engine::request_backend_swap), but the
            // 03_studio sample boots its own device + swapchain manually
            // instead of going through Engine. To make this command live,
            // either:
            //   (a) Migrate this sample to use cardinal::engine::Engine
            //       (one Application subclass, ~30 lines), then forward
            //       this command to engine.request_backend_swap(target).
            //   (b) Add scene::Mesh::release_gpu() + reupload(Device&)
            //       so we can iterate the scene + asset catalog and
            //       re-upload every mesh after rebuilding the device.
            //
            // The Studio side is already swap-ready: Studio::reinit(...)
            // tears down the ImGui backend on the OLD device and rebinds
            // onto the new one, preserving every UI state (windows, dock
            // layout, console scrollback, registered tools) across the
            // swap. The Engine's perform_backend_swap orchestrates the
            // full sequence (drain GPU, before-hook, recreate device +
            // swapchain + pipelines, Studio::reinit, after-hook).
            out("Hot-swap is wired in cardinal::engine::Engine "
                "(set_backend_swap_hooks + request_backend_swap), but this");
            out("sample doesn't use Engine — it boots manually. "
                "To make this command live: migrate to Engine, or hook");
            out("scene::Mesh::reupload + dup the orchestration here.");
            out("requested target: %s", argv[1].c_str());
        });
}

}  // namespace sample_studio
