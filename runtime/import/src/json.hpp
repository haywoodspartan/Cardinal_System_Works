#pragma once

// =============================================================================
// Cardinal — minimal JSON value + recursive-descent parser (src-private).
//
// Hand-rolled, zero-dep (no nlohmann / rapidjson), same ethos as the rest of
// the importer. Originally lived in import_gltf.cpp's anonymous namespace;
// lifted here so BOTH the glTF backend and the Megascans manifest backend
// parse JSON through ONE implementation instead of duplicating it.
//
// This is a PRIVATE source header (under src/, not include/) — it is part of
// cardinal::import's internals, not its public surface. It lives in
// cardinal::import::detail so the two TUs share the exact same type with no
// ODR risk (all members are in-class / implicitly inline).
//
// Scope: just enough JSON for glTF documents + Bridge manifests — objects,
// arrays, strings (with \uXXXX basic-plane escapes), numbers, bool, null.
// Robust to truncation (the `ok` flag goes false; readers degrade, never
// crash) since both callers parse untrusted, externally-authored files.
// =============================================================================

#include <cardinal/core/types.hpp>        // cardinal::string / u-ints
#include <cardinal/core/containers.hpp>   // cardinal::vector / unordered_map
#include <cardinal/core/cstdlib.hpp>      // cardinal::strtod / strtol
#include <cardinal/core/cstring.hpp>      // cardinal::strncmp
#include <cardinal/core/utility.hpp>      // cardinal::move

namespace cardinal::import::detail {

// ---------------------------------------------------------------------------
// JSON value.
// ---------------------------------------------------------------------------
struct JVal {
    enum class T { Null, Bool, Num, Str, Arr, Obj } t{T::Null};
    bool                                       b{false};
    double                                     n{0.0};
    cardinal::string                                s;
    cardinal::vector<JVal>                          arr;
    cardinal::unordered_map<cardinal::string, JVal>      obj;

    bool is_obj() const { return t == T::Obj; }
    bool is_arr() const { return t == T::Arr; }
    const JVal* find(const char* k) const {
        if (t != T::Obj) return nullptr;
        auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    }
    double num(double dflt = 0.0) const { return t == T::Num ? n : dflt; }
    int    inum(int dflt = -1)    const { return t == T::Num ? static_cast<int>(n) : dflt; }
    cardinal::string str() const { return t == T::Str ? s : cardinal::string{}; }
};

// ---------------------------------------------------------------------------
// Recursive-descent parser.
// ---------------------------------------------------------------------------
struct JParser {
    const char* p;
    const char* e;
    bool ok{true};

    void ws() { while (p < e && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p; }

    JVal parse() { ws(); return value(); }

    JVal value() {
        ws();
        if (p >= e) { ok = false; return {}; }
        switch (*p) {
            case '{': return object();
            case '[': return array();
            case '"': { JVal v; v.t = JVal::T::Str; v.s = string(); return v; }
            case 't': case 'f': return boolean();
            case 'n': p += (e - p >= 4) ? 4 : (e - p); return {}; // null
            default:  return number();
        }
    }
    JVal object() {
        JVal v; v.t = JVal::T::Obj; ++p; ws();
        if (p < e && *p == '}') { ++p; return v; }
        while (p < e) {
            ws();
            cardinal::string key = string();
            ws();
            if (p < e && *p == ':') ++p;
            v.obj.emplace(cardinal::move(key), value());
            ws();
            if (p < e && *p == ',') { ++p; continue; }
            if (p < e && *p == '}') { ++p; break; }
            ok = false; break;
        }
        return v;
    }
    JVal array() {
        JVal v; v.t = JVal::T::Arr; ++p; ws();
        if (p < e && *p == ']') { ++p; return v; }
        while (p < e) {
            v.arr.push_back(value());
            ws();
            if (p < e && *p == ',') { ++p; continue; }
            if (p < e && *p == ']') { ++p; break; }
            ok = false; break;
        }
        return v;
    }
    cardinal::string string() {
        cardinal::string out;
        if (p >= e || *p != '"') { ok = false; return out; }
        ++p;
        while (p < e && *p != '"') {
            char c = *p++;
            if (c == '\\' && p < e) {
                char x = *p++;
                switch (x) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case '/': out += '/';  break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        // Basic-plane only; good enough for asset names.
                        if (e - p >= 4) {
                            int cp = static_cast<int>(
                                cardinal::strtol(cardinal::string(p, p+4).c_str(),
                                            nullptr, 16));
                            p += 4;
                            if (cp < 0x80) out += static_cast<char>(cp);
                            else if (cp < 0x800) {
                                out += static_cast<char>(0xC0 | (cp >> 6));
                                out += static_cast<char>(0x80 | (cp & 0x3F));
                            } else {
                                out += static_cast<char>(0xE0 | (cp >> 12));
                                out += static_cast<char>(0x80 | ((cp>>6)&0x3F));
                                out += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                        }
                        break;
                    }
                    default: out += x; break;
                }
            } else {
                out += c;
            }
        }
        if (p < e) ++p;  // closing quote
        return out;
    }
    JVal boolean() {
        JVal v; v.t = JVal::T::Bool;
        if (e - p >= 4 && cardinal::strncmp(p, "true", 4) == 0)  { v.b=true;  p+=4; }
        else if (e - p >= 5 && cardinal::strncmp(p,"false",5)==0){ v.b=false; p+=5; }
        else ok = false;
        return v;
    }
    JVal number() {
        const char* s = p;
        while (p < e && (*p=='-'||*p=='+'||*p=='.'||*p=='e'||*p=='E'||
                         (*p>='0'&&*p<='9'))) ++p;
        JVal v; v.t = JVal::T::Num;
        v.n = cardinal::strtod(cardinal::string(s, p).c_str(), nullptr);
        return v;
    }
};

}  // namespace cardinal::import::detail
