// =============================================================================
// Cardinal — engine console implementation.
// =============================================================================
#include <cardinal/console/console.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::min/sort/unique/clamp
#include <cardinal/core/cctype.hpp>      // cardinal::tolower/isspace
#include <cardinal/core/cmath.hpp>       // cardinal::isfinite
#include <cardinal/core/cstdarg.hpp>     // cardinal::va_list (+ va_* macros)
#include <cardinal/core/cstdio.hpp>      // cardinal::vsnprintf
#include <cardinal/core/cstdlib.hpp>     // cardinal::strtoll/strtod
#include <cardinal/core/cstring.hpp>

namespace cardinal::console {

// ----------------------------------------------------------------------------
// Output — printf-style write that funnels each line through the user sink.
// ----------------------------------------------------------------------------
void Output::operator()(const char* fmt, ...) {
    if (!fn_ || fmt == nullptr) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    const int n = cardinal::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    fn_(cardinal::string(buf, static_cast<usize>(cardinal::min<int>(n, sizeof(buf) - 1))));
}

// ----------------------------------------------------------------------------
// Helpers — name normalisation + line tokenisation.
// ----------------------------------------------------------------------------
namespace {

cardinal::string lower(cardinal::string s) {
    for (auto& c : s) c = static_cast<char>(cardinal::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool ieq(const cardinal::string& a, const cardinal::string& b) noexcept {
    if (a.size() != b.size()) return false;
    for (usize i = 0; i < a.size(); ++i) {
        if (cardinal::tolower(static_cast<unsigned char>(a[i])) !=
            cardinal::tolower(static_cast<unsigned char>(b[i]))) return false;
    }
    return true;
}

// Split a console line into tokens. "double quoted strings" survive as one
// token; everything else is whitespace-separated. Backslash-escapes are not
// supported (UE5 does the same — quotes are literal containers, no escapes).
cardinal::vector<cardinal::string> tokenize(const cardinal::string& s) {
    cardinal::vector<cardinal::string> out;
    cardinal::string cur;
    bool in_quote = false;
    for (char c : s) {
        if (in_quote) {
            if (c == '"') { out.push_back(cardinal::move(cur)); cur.clear(); in_quote = false; }
            else          { cur.push_back(c); }
        } else if (c == '"') {
            if (!cur.empty()) { out.push_back(cardinal::move(cur)); cur.clear(); }
            in_quote = true;
        } else if (cardinal::isspace(static_cast<unsigned char>(c))) {
            if (!cur.empty()) { out.push_back(cardinal::move(cur)); cur.clear(); }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cardinal::move(cur));
    return out;
}

bool parse_bool(const cardinal::string& s, bool& out) noexcept {
    const cardinal::string l = lower(s);
    if (l == "1" || l == "true" || l == "on"  || l == "yes")  { out = true;  return true; }
    if (l == "0" || l == "false"|| l == "off" || l == "no")   { out = false; return true; }
    return false;
}

bool parse_int(const cardinal::string& s, i64& out) noexcept {
    char* end = nullptr;
    const long long v = cardinal::strtoll(s.c_str(), &end, 0);
    if (end == s.c_str()) return false;
    out = static_cast<i64>(v);
    return true;
}

bool parse_float(const cardinal::string& s, f64& out) noexcept {
    char* end = nullptr;
    const double v = cardinal::strtod(s.c_str(), &end);
    // strtod (per the C standard) also parses "nan"/"inf"/"infinity". A
    // NaN would then DEFEAT apply_cvar's [min,max] clamp — `v < min` and
    // `v > max` are both false for NaN (unordered) — writing NaN straight
    // into live engine state. Treat any non-finite (or unparseable) input
    // as a parse failure: caller keeps the old value and reports an error.
    if (end == s.c_str() || !cardinal::isfinite(v)) return false;
    out = static_cast<f64>(v);
    return true;
}

}  // namespace

// ----------------------------------------------------------------------------
// Registry — singleton + (un)registration + lookup.
// ----------------------------------------------------------------------------
Registry& Registry::instance() noexcept {
    static Registry r;
    return r;
}

bool Registry::register_cvar(CVar cv) {
    if (cv.name.empty()) return false;
    if (find_cvar(cv.name) != nullptr) return false;
    if (find_command(cv.name) != nullptr) return false;
    cvars_.push_back(cardinal::move(cv));
    return true;
}

bool Registry::register_command(CCommand cc) {
    if (cc.name.empty() || cc.fn == nullptr) return false;
    if (find_cvar(cc.name) != nullptr) return false;
    if (find_command(cc.name) != nullptr) return false;
    commands_.push_back(cardinal::move(cc));
    return true;
}

void Registry::unregister(const cardinal::string& name) {
    for (auto it = cvars_.begin(); it != cvars_.end(); ++it) {
        if (ieq(it->name, name)) { cvars_.erase(it); return; }
    }
    for (auto it = commands_.begin(); it != commands_.end(); ++it) {
        if (ieq(it->name, name)) { commands_.erase(it); return; }
    }
}

void Registry::clear() {
    cvars_.clear();
    commands_.clear();
}

const CVar* Registry::find_cvar(const cardinal::string& name) const noexcept {
    for (const auto& c : cvars_) if (ieq(c.name, name)) return &c;
    return nullptr;
}

const CCommand* Registry::find_command(const cardinal::string& name) const noexcept {
    for (const auto& c : commands_) if (ieq(c.name, name)) return &c;
    return nullptr;
}

cardinal::vector<const CVar*> Registry::all_cvars() const {
    cardinal::vector<const CVar*> out;
    out.reserve(cvars_.size());
    for (const auto& c : cvars_) out.push_back(&c);
    cardinal::sort(out.begin(), out.end(),
              [](const CVar* a, const CVar* b) { return a->name < b->name; });
    return out;
}

cardinal::vector<const CCommand*> Registry::all_commands() const {
    cardinal::vector<const CCommand*> out;
    out.reserve(commands_.size());
    for (const auto& c : commands_) out.push_back(&c);
    cardinal::sort(out.begin(), out.end(),
              [](const CCommand* a, const CCommand* b) { return a->name < b->name; });
    return out;
}

cardinal::vector<cardinal::string> Registry::complete(const cardinal::string& prefix,
                                            usize max_results) const {
    cardinal::vector<cardinal::string> out;
    out.reserve(max_results);
    const cardinal::string p = lower(prefix);
    auto consider = [&](const cardinal::string& name) {
        if (out.size() >= max_results) return;
        const cardinal::string ln = lower(name);
        if (ln.size() < p.size()) return;
        if (ln.compare(0, p.size(), p) != 0) return;
        out.push_back(name);
    };
    for (const auto& c : cvars_)    consider(c.name);
    for (const auto& c : commands_) consider(c.name);
    cardinal::sort(out.begin(), out.end());
    out.erase(cardinal::unique(out.begin(), out.end()), out.end());
    if (out.size() > max_results) out.resize(max_results);
    return out;
}

// ----------------------------------------------------------------------------
// CVar print + apply — used by execute() for the `<cvar>` and `<cvar> <value>`
// forms. Pulled out so the same code services interactive use AND scripted
// drive (e.g. a config file replays each line through Registry::execute).
// ----------------------------------------------------------------------------
namespace {

void print_cvar(const CVar& cv, Output& out) {
    switch (cv.type) {
        case CVarType::Bool:
            out("%s = %s   (bool) — %s",
                cv.name.c_str(),
                (cv.get_bool && cv.get_bool()) ? "true" : "false",
                cv.help.c_str());
            break;
        case CVarType::Int:
            out("%s = %lld   (int [%lld..%lld]) — %s",
                cv.name.c_str(),
                cv.get_int ? static_cast<long long>(cv.get_int()) : 0LL,
                static_cast<long long>(cv.min),
                static_cast<long long>(cv.max),
                cv.help.c_str());
            break;
        case CVarType::Float:
            out("%s = %.6g   (float [%.4g..%.4g]) — %s",
                cv.name.c_str(),
                cv.get_float ? cv.get_float() : 0.0,
                cv.min, cv.max,
                cv.help.c_str());
            break;
        case CVarType::String:
            out("%s = \"%s\"   (string) — %s",
                cv.name.c_str(),
                cv.get_string ? cv.get_string().c_str() : "",
                cv.help.c_str());
            break;
    }
}

bool apply_cvar(const CVar& cv, const cardinal::string& raw_value, Output& out) {
    switch (cv.type) {
        case CVarType::Bool: {
            bool b{};
            if (!parse_bool(raw_value, b)) {
                out("error: %s expects bool (1/0/true/false/on/off/yes/no)",
                    cv.name.c_str());
                return false;
            }
            if (cv.set_bool) cv.set_bool(b);
            return true;
        }
        case CVarType::Int: {
            i64 v{};
            if (!parse_int(raw_value, v)) {
                out("error: %s expects integer", cv.name.c_str());
                return false;
            }
            const i64 lo = static_cast<i64>(cv.min);
            const i64 hi = static_cast<i64>(cv.max);
            if (lo < hi && (v < lo || v > hi)) {
                out("warning: clamping %lld to [%lld..%lld]",
                    static_cast<long long>(v),
                    static_cast<long long>(lo),
                    static_cast<long long>(hi));
                v = cardinal::clamp(v, lo, hi);
            }
            if (cv.set_int) cv.set_int(v);
            return true;
        }
        case CVarType::Float: {
            f64 v{};
            if (!parse_float(raw_value, v)) {
                out("error: %s expects number", cv.name.c_str());
                return false;
            }
            if (cv.min < cv.max && (v < cv.min || v > cv.max)) {
                out("warning: clamping %.6g to [%.4g..%.4g]",
                    v, cv.min, cv.max);
                v = cardinal::clamp(v, cv.min, cv.max);
            }
            if (cv.set_float) cv.set_float(v);
            return true;
        }
        case CVarType::String: {
            if (cv.set_string) cv.set_string(raw_value);
            return true;
        }
    }
    return false;
}

}  // namespace

bool Registry::execute(const cardinal::string& input, Output& out) {
    auto argv = tokenize(input);
    if (argv.empty()) return false;

    const cardinal::string& head = argv[0];

    // Built-in `help` / `?` — list categories or describe one entry. Handled
    // before lookup so users can `help` even when nothing else is bound.
    if (ieq(head, "help") || head == "?") {
        if (argv.size() == 1) {
            out("Console — %zu cvars, %zu commands. Use `help <name>` for details, `list` for all names.",
                cvars_.size(), commands_.size());
            out("Built-ins: help, ?, list, list cvars, list commands");
            return true;
        }
        const cardinal::string& q = argv[1];
        if (const auto* cv = find_cvar(q)) {
            print_cvar(*cv, out);
            return true;
        }
        if (const auto* cc = find_command(q)) {
            out("%s — %s", cc->name.c_str(), cc->help.c_str());
            return true;
        }
        out("help: '%s' not found", q.c_str());
        return true;
    }

    // Built-in `list [cvars|commands]` — dump every registered name.
    if (ieq(head, "list")) {
        const bool want_cvars   = (argv.size() < 2) || ieq(argv[1], "cvars");
        const bool want_cmds    = (argv.size() < 2) || ieq(argv[1], "commands");
        if (want_cvars) {
            for (const auto* cv : all_cvars()) print_cvar(*cv, out);
        }
        if (want_cmds) {
            for (const auto* cc : all_commands()) {
                out("%s — %s", cc->name.c_str(), cc->help.c_str());
            }
        }
        return true;
    }

    // CVar branch.
    if (const CVar* cv = find_cvar(head)) {
        if (argv.size() == 1) {
            print_cvar(*cv, out);
        } else {
            // Re-join args 1..N so set_string preserves the user's spaces.
            cardinal::string value;
            for (usize i = 1; i < argv.size(); ++i) {
                if (i > 1) value.push_back(' ');
                value += argv[i];
            }
            if (apply_cvar(*cv, value, out)) {
                print_cvar(*cv, out);
            }
        }
        return true;
    }

    // Command branch.
    if (const CCommand* cc = find_command(head)) {
        cc->fn(argv, out);
        return true;
    }

    // Unknown — let the caller fall through to a scripting engine.
    return false;
}

}  // namespace cardinal::console
