// =============================================================================
// Cardinal — Lua engine implementation (C API, no sol2).
//
// Hosts a single lua_State with engine bindings, a REPL helper, and a
// debugger surface (breakpoints, single-step, locals/upvalues/stack
// inspection) plus a function-call/return tracer wired into the global
// trace timeline. Both the debugger and tracer share one lua_sethook
// installed on the active state.
// =============================================================================
#include <cardinal/script/engine.hpp>

#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>
#include <cardinal/trace/timeline.hpp>

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#if CARDINAL_PLATFORM_WINDOWS
    #include <Windows.h>
#endif

#include <cardinal/core/std/chrono.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/cstdio.hpp>
#include <cardinal/core/std/cstring.hpp>
#include <cardinal/core/std/fstream.hpp>
#include <cardinal/core/std/sstream.hpp>
#include <cardinal/core/std/thread.hpp>
#include <cardinal/core/std/utility.hpp>

namespace cardinal::script {

namespace {

// ---- engine bindings (C++ → Lua) ------------------------------------------
//
// Calling convention:
//   cardinal.log_info("category", "message")  — explicit category
//   cardinal.log_info("message")              — implicit "script" category
namespace {
void log_dispatch(lua_State* L, void (*sink)(const char*, const char*, ...)) {
    const int argc = lua_gettop(L);
    if (argc <= 0) { sink("script", "%s", ""); return; }
    if (argc == 1) {
        const char* m = luaL_tolstring(L, 1, nullptr);
        sink("script", "%s", m ? m : "");
        return;
    }
    const char* cat = luaL_tolstring(L, 1, nullptr);
    const char* msg = luaL_tolstring(L, 2, nullptr);
    sink(cat ? cat : "script", "%s", msg ? msg : "");
}
}
int l_log_info (lua_State* L) { log_dispatch(L, &cardinal::log::infof);  return 0; }
int l_log_warn (lua_State* L) { log_dispatch(L, &cardinal::log::warnf);  return 0; }
int l_log_error(lua_State* L) { log_dispatch(L, &cardinal::log::errorf); return 0; }

void register_bindings(lua_State* L) {
    lua_newtable(L);
    lua_pushcfunction(L, l_log_info);  lua_setfield(L, -2, "log_info");
    lua_pushcfunction(L, l_log_warn);  lua_setfield(L, -2, "log_warn");
    lua_pushcfunction(L, l_log_error); lua_setfield(L, -2, "log_error");
    lua_setglobal(L, "cardinal");
}

cardinal::string pop_error(lua_State* L) {
    const char* msg = lua_tostring(L, -1);
    cardinal::string out = msg ? msg : "(no error message)";
    lua_pop(L, 1);
    return out;
}

// Return the canonical "source name" used by breakpoint matching. Lua
// stores `lua_Debug::source` like "@path/to/file.lua" for files and
// "=[chunk name]" for in-memory chunks; we strip the leading sigil.
cardinal::string canonical_source(const lua_Debug& ar) {
    if (ar.source == nullptr) return {};
    if (ar.source[0] == '@' || ar.source[0] == '=') return ar.source + 1;
    return ar.source;
}

class EngineImpl final : public Engine {
public:
    EngineImpl() {
        L_ = luaL_newstate();
        if (L_) {
            luaL_openlibs(L_);
            register_bindings(L_);
            // Stash a back-pointer so the static hook can recover us.
            lua_pushlightuserdata(L_, this);
            lua_setfield(L_, LUA_REGISTRYINDEX, "cardinal_script_engine");
            cardinal::log::infof("script", "Lua engine initialised (%s)", LUA_RELEASE);
        } else {
            cardinal::log::errorf("script", "luaL_newstate failed");
        }
    }
    ~EngineImpl() override { if (L_) lua_close(L_); }

    cardinal::string run_string(const char* code, const char* chunk_name) override {
        if (L_ == nullptr) return "Lua engine not initialised";
        if (luaL_loadbuffer(L_, code, cardinal::strlen(code),
                            chunk_name ? chunk_name : "=chunk") != LUA_OK) {
            return pop_error(L_);
        }
        if (lua_pcall(L_, 0, 0, 0) != LUA_OK) {
            return pop_error(L_);
        }
        return {};
    }

    cardinal::string repl_eval(const char* line) override {
        if (L_ == nullptr) return "(no engine)";
        if (line == nullptr || *line == '\0') return {};

        cardinal::string expr = "return ("; expr += line; expr += ")";
        const int top0 = lua_gettop(L_);
        bool returned_value = false;
        if (luaL_loadbuffer(L_, expr.data(), expr.size(), "=repl") == LUA_OK &&
            lua_pcall(L_, 0, LUA_MULTRET, 0) == LUA_OK)
        {
            returned_value = true;
        } else {
            lua_settop(L_, top0);
            if (luaL_loadbuffer(L_, line, cardinal::strlen(line), "=repl") != LUA_OK) {
                return pop_error(L_);
            }
            if (lua_pcall(L_, 0, LUA_MULTRET, 0) != LUA_OK) {
                return pop_error(L_);
            }
        }

        const int n = lua_gettop(L_) - top0;
        if (n <= 0) return returned_value ? cardinal::string("nil") : cardinal::string{};

        cardinal::string out;
        for (int i = 0; i < n; ++i) {
            const char* s = luaL_tolstring(L_, top0 + 1 + i, nullptr);
            if (i) out += "\t";
            out += s ? s : "nil";
            lua_pop(L_, 1);
        }
        lua_settop(L_, top0);
        return out;
    }

    cardinal::string run_file(const char* path) override {
        if (L_ == nullptr) return "Lua engine not initialised";
        cardinal::ifstream f(path);
        if (!f.good()) {
            cardinal::string err = "cannot open: ";
            err += path ? path : "(null)";
            return err;
        }
        cardinal::stringstream ss; ss << f.rdbuf();
        cardinal::string src = ss.str();
        return run_string(src.c_str(), path);
    }

    // ---- debugger surface ---------------------------------------------
    void set_debug_enabled(bool on) override {
        if (L_ == nullptr || debug_enabled_ == on) return;
        debug_enabled_ = on;
        if (on) {
            // CALL+RET → trace timeline, LINE → breakpoint check / step
            lua_sethook(L_, &EngineImpl::lua_hook_thunk,
                        LUA_MASKCALL | LUA_MASKRET | LUA_MASKLINE, 0);
        } else {
            lua_sethook(L_, nullptr, 0, 0);
        }
    }
    bool debug_enabled() const noexcept override { return debug_enabled_; }

    void add_breakpoint(const char* source, u32 line) override {
        if (source == nullptr || line == 0) return;
        breakpoints_.insert({source, line});
    }
    void remove_breakpoint(const char* source, u32 line) override {
        breakpoints_.erase({source ? source : "", line});
    }
    void clear_breakpoints() override { breakpoints_.clear(); }
    cardinal::vector<cardinal::pair<cardinal::string, u32>> breakpoints() const override {
        return { breakpoints_.begin(), breakpoints_.end() };
    }

    DebugState state() const noexcept override { return state_; }

    void resume(StepMode mode) override {
        if (state_ == DebugState::Idle) return;
        step_mode_       = mode;
        step_base_depth_ = paused_call_depth_;
        state_           = (mode == StepMode::Continue) ? DebugState::Idle : DebugState::Stepping;
    }

    cardinal::vector<DebugFrame> stack() const override { return paused_stack_; }
    cardinal::vector<DebugVar>   locals() const override { return paused_locals_; }
    cardinal::vector<DebugVar>   upvalues() const override { return paused_upvalues_; }

private:
    static EngineImpl* from_state(lua_State* L) {
        lua_getfield(L, LUA_REGISTRYINDEX, "cardinal_script_engine");
        auto* self = static_cast<EngineImpl*>(lua_touserdata(L, -1));
        lua_pop(L, 1);
        return self;
    }

    static void lua_hook_thunk(lua_State* L, lua_Debug* ar) {
        if (auto* self = from_state(L)) self->on_hook(L, ar);
    }

    void on_hook(lua_State* L, lua_Debug* ar) {
        // Trace events for CALL/RET — push to the global timeline.
        if (ar->event == LUA_HOOKCALL || ar->event == LUA_HOOKTAILCALL) {
            ++call_depth_;
            lua_getinfo(L, "nS", ar);
            const char* name = ar->name ? ar->name :
                              (ar->what && cardinal::strcmp(ar->what, "main") == 0 ? "<main>" : "?");
            cardinal::trace::Timeline::instance().push(
                name, "lua", cardinal::trace::EventKind::Call, call_depth_);
            return;
        }
        if (ar->event == LUA_HOOKRET) {
            lua_getinfo(L, "nS", ar);
            const char* name = ar->name ? ar->name : "?";
            cardinal::trace::Timeline::instance().push(
                name, "lua", cardinal::trace::EventKind::Return, call_depth_);
            if (call_depth_ > 0) --call_depth_;
            // Step Out: resume when we leave the frame we were in.
            if (state_ == DebugState::Stepping && step_mode_ == StepMode::Out &&
                call_depth_ < step_base_depth_) {
                pause_here(L, ar);
            }
            return;
        }

        // LINE events drive both step + breakpoint matching.
        if (ar->event != LUA_HOOKLINE) return;

        // Service step requests first.
        if (state_ == DebugState::Stepping) {
            switch (step_mode_) {
                case StepMode::In:
                    pause_here(L, ar); return;
                case StepMode::Over:
                    if (call_depth_ <= step_base_depth_) { pause_here(L, ar); return; }
                    break;
                case StepMode::Out:   /* handled in RET */ break;
                case StepMode::Continue: state_ = DebugState::Idle; break;
            }
        }

        // Breakpoint match.
        if (!breakpoints_.empty()) {
            lua_getinfo(L, "Sl", ar);
            const cardinal::string src = canonical_source(*ar);
            if (breakpoints_.count({src, static_cast<u32>(ar->currentline)})) {
                pause_here(L, ar);
            }
        }
    }

    // Block the VM until resume() is called from the main thread. We do
    // this by spinning + lua_sethook hint check. The Studio panel is on
    // the main thread, where `resume()` mutates state_ → step_mode_ →
    // we release.
    void pause_here(lua_State* L, lua_Debug* ar) {
        // Snapshot context for the panel.
        snapshot_context(L, ar);
        state_             = DebugState::AtBreakpoint;
        paused_call_depth_ = call_depth_;

        cardinal::log::infof("script",
            "paused at %s:%d  (resume from Script Debugger panel)",
            paused_stack_.empty() ? "?" : paused_stack_[0].source.c_str(),
            paused_stack_.empty() ? 0   : paused_stack_[0].current_line);

        // Spin-wait. Sleep briefly between checks so we don't pin a core.
        while (state_ == DebugState::AtBreakpoint) {
            cardinal::this_thread::sleep_for(cardinal::chrono::milliseconds(2));
        }
    }

    void snapshot_context(lua_State* L, lua_Debug* /*ar*/) {
        paused_stack_.clear();
        paused_locals_.clear();
        paused_upvalues_.clear();

        lua_Debug d{};
        for (int level = 0; lua_getstack(L, level, &d); ++level) {
            lua_getinfo(L, "Snl", &d);
            DebugFrame f;
            f.source       = canonical_source(d);
            f.current_line = static_cast<u32>(d.currentline);
            f.what         = d.what  ? d.what  : "";
            f.name         = d.name  ? d.name  : "";
            paused_stack_.push_back(cardinal::move(f));
        }
        if (!paused_stack_.empty() && lua_getstack(L, 0, &d)) {
            // Locals at frame 0.
            for (int i = 1;; ++i) {
                const char* n = lua_getlocal(L, &d, i);
                if (n == nullptr) break;
                paused_locals_.push_back(make_var(L, n));
                lua_pop(L, 1);
            }
            // Upvalues at frame 0.
            lua_getinfo(L, "f", &d);  // pushes the function
            for (int i = 1;; ++i) {
                const char* n = lua_getupvalue(L, -1, i);
                if (n == nullptr) break;
                paused_upvalues_.push_back(make_var(L, n));
                lua_pop(L, 1);
            }
            lua_pop(L, 1);            // pop the function
        }
    }

    DebugVar make_var(lua_State* L, const char* name) {
        DebugVar v;
        v.name = name ? name : "";
        v.type = lua_typename(L, lua_type(L, -1));
        const char* s = luaL_tolstring(L, -1, nullptr);
        v.value = s ? s : "";
        if (v.value.size() > 64) v.value.resize(64);
        lua_pop(L, 1);   // pop the tostring copy
        return v;
    }

    lua_State* L_{nullptr};

    // Debugger state
    bool                                            debug_enabled_{false};
    cardinal::set<cardinal::pair<cardinal::string, u32>>           breakpoints_;
    DebugState                                      state_{DebugState::Idle};
    StepMode                                        step_mode_{StepMode::Continue};
    u32                                             call_depth_{0};
    u32                                             paused_call_depth_{0};
    u32                                             step_base_depth_{0};

    // Snapshot for the panel
    cardinal::vector<DebugFrame>                         paused_stack_;
    cardinal::vector<DebugVar>                           paused_locals_;
    cardinal::vector<DebugVar>                           paused_upvalues_;
};

}  // namespace

cardinal::unique_ptr<Engine> Engine::create() { return cardinal::make_unique<EngineImpl>(); }

}  // namespace cardinal::script
