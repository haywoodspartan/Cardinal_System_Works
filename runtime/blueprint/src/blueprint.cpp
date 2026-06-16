// =============================================================================
// Cardinal Blueprint — graph <-> source round-trip engine.
// =============================================================================
#include <cardinal/blueprint/blueprint.hpp>

#include <cardinal/core/std/cstdio.hpp>     // snprintf
#include <cardinal/core/std/cstdlib.hpp>    // strtod
#include <cardinal/core/std/cstring.hpp>

namespace cardinal::blueprint {

// =============================================================================
// Codegen : Graph -> canonical source text
// =============================================================================
namespace {

// Readable number formatting: integral values print without a decimal point
// ("5"), others with the shortest %g form ("2.5"). Round-trips through strtod.
cardinal::string fmt_num(f64 v) {
    char buf[64];
    const f64 r = static_cast<f64>(static_cast<i64>(v));
    if (r == v && v < 9.0e15 && v > -9.0e15)
        cardinal::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
    else
        cardinal::snprintf(buf, sizeof(buf), "%g", v);
    return cardinal::string(buf);
}

void emit_arg(const Arg& a, cardinal::string& out) {
    if (a.is_literal) out += fmt_num(a.literal);
    else              out += a.var;
}

void emit_args(const cardinal::vector<Arg>& args, cardinal::string& out) {
    out += '(';
    for (usize i = 0; i < args.size(); ++i) {
        if (i) out += ", ";
        emit_arg(args[i], out);
    }
    out += ')';
}

}  // namespace

cardinal::string generate_source(const Graph& g) {
    cardinal::string out;
    for (usize fi = 0; fi < g.funcs.size(); ++fi) {
        const Function& f = g.funcs[fi];
        if (fi) out += '\n';
        out += "func ";
        out += f.name;
        out += '(';
        for (usize i = 0; i < f.params.size(); ++i) {
            if (i) out += ", ";
            out += f.params[i].name;
            if (!f.params[i].type.empty()) { out += ": "; out += f.params[i].type; }
        }
        out += ") {\n";
        for (const Node& n : f.nodes) {
            out += "  ";
            switch (n.kind) {
            case NodeKind::Literal:
                out += "let "; out += n.out; out += " = "; out += fmt_num(n.literal);
                break;
            case NodeKind::Return:
                out += "return";
                if (!n.args.empty()) { out += ' '; emit_arg(n.args[0], out); }
                break;
            case NodeKind::Call:
                if (!n.out.empty()) { out += "let "; out += n.out; out += " = "; }
                out += n.fn;
                emit_args(n.args, out);
                break;
            }
            out += '\n';
        }
        out += "}\n";
    }
    return out;
}

// =============================================================================
// Lexer
// =============================================================================
namespace {

enum class Tok : u32 {
    End, Error,
    Ident, Number,
    LParen, RParen, LBrace, RBrace, Comma, Equals, Colon,
    KwFunc, KwLet, KwReturn,
};

struct Token {
    Tok              kind{Tok::End};
    cardinal::string text;   // Ident text / raw number text
    f64              num{0.0};
    u32              line{1};
};

class Lexer {
public:
    explicit Lexer(const cardinal::string& src) : s_(src) {}

    Token next() {
        skip_ws_and_comments();
        Token t;
        t.line = line_;
        if (pos_ >= s_.size()) { t.kind = Tok::End; return t; }

        const char c = s_[pos_];
        switch (c) {
        case '(': ++pos_; t.kind = Tok::LParen; return t;
        case ')': ++pos_; t.kind = Tok::RParen; return t;
        case '{': ++pos_; t.kind = Tok::LBrace; return t;
        case '}': ++pos_; t.kind = Tok::RBrace; return t;
        case ',': ++pos_; t.kind = Tok::Comma;  return t;
        case '=': ++pos_; t.kind = Tok::Equals; return t;
        case ':': ++pos_; t.kind = Tok::Colon;  return t;
        default: break;
        }

        if (is_num_start(c)) return lex_number(t);
        if (is_ident_start(c)) return lex_ident(t);

        t.kind = Tok::Error;
        t.text = "unexpected character";
        return t;
    }

    u32 line() const noexcept { return line_; }

private:
    static bool is_ident_start(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    }
    static bool is_ident(char c) { return is_ident_start(c) || (c >= '0' && c <= '9'); }
    static bool is_digit(char c) { return c >= '0' && c <= '9'; }
    bool is_num_start(char c) const {
        return is_digit(c) ||
               ((c == '-' || c == '+' || c == '.') && pos_ + 1 < s_.size() &&
                is_digit(s_[pos_ + 1]));
    }

    void skip_ws_and_comments() {
        for (;;) {
            while (pos_ < s_.size()) {
                const char c = s_[pos_];
                if (c == '\n') { ++line_; ++pos_; }
                else if (c == ' ' || c == '\t' || c == '\r') { ++pos_; }
                else break;
            }
            // line comment: // ... to EOL
            if (pos_ + 1 < s_.size() && s_[pos_] == '/' && s_[pos_ + 1] == '/') {
                pos_ += 2;
                while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
                continue;
            }
            break;
        }
    }

    Token lex_number(Token& t) {
        const usize start = pos_;
        if (s_[pos_] == '-' || s_[pos_] == '+') ++pos_;
        while (pos_ < s_.size() && (is_digit(s_[pos_]) || s_[pos_] == '.' ||
               s_[pos_] == 'e' || s_[pos_] == 'E' ||
               ((s_[pos_] == '-' || s_[pos_] == '+') &&
                (s_[pos_-1] == 'e' || s_[pos_-1] == 'E')))) ++pos_;
        t.kind = Tok::Number;
        t.text = s_.substr(start, pos_ - start);
        t.num  = cardinal::strtod(t.text.c_str(), nullptr);
        return t;
    }

    Token lex_ident(Token& t) {
        const usize start = pos_;
        while (pos_ < s_.size() && is_ident(s_[pos_])) ++pos_;
        t.text = s_.substr(start, pos_ - start);
        if      (t.text == "func")   t.kind = Tok::KwFunc;
        else if (t.text == "let")    t.kind = Tok::KwLet;
        else if (t.text == "return") t.kind = Tok::KwReturn;
        else                         t.kind = Tok::Ident;
        return t;
    }

    const cardinal::string& s_;
    usize pos_{0};
    u32   line_{1};
};

// =============================================================================
// Parser (recursive descent) : source -> Graph, flattening nested calls.
// =============================================================================
class Parser {
public:
    Parser(const cardinal::string& src, cardinal::string* err)
        : lex_(src), err_(err) { advance(); }

    bool parse(Graph& g) {
        while (cur_.kind != Tok::End) {
            if (cur_.kind == Tok::Error) return fail(cur_.text.c_str());
            if (cur_.kind != Tok::KwFunc) return fail("expected 'func'");
            Function f;
            if (!parse_func(f)) return false;
            g.funcs.push_back(cardinal::move(f));
        }
        return true;
    }

private:
    // --- primary expression -> (arg, node_idx). node_idx >= 0 if it produced a
    //     call node (so a `let`/statement can rename/clear it). ---------------
    struct Primary { Arg arg; isize node_idx{-1}; };

    bool parse_primary(Primary& p) {
        if (cur_.kind == Tok::Number) {
            p.arg = Arg::lit(cur_.num);
            advance();
            return true;
        }
        if (cur_.kind == Tok::Ident) {
            cardinal::string name = cur_.text;
            advance();
            if (cur_.kind == Tok::LParen) {           // a call
                isize idx = -1;
                if (!parse_call(cardinal::move(name), idx)) return false;
                p.node_idx = idx;
                p.arg = Arg::ref(fn_->nodes[static_cast<usize>(idx)].out);
                return true;
            }
            p.arg = Arg::ref(cardinal::move(name));   // a wire / variable ref
            return true;
        }
        return fail("expected value, variable, or call");
    }

    // Consume '(' args ')'. Creates a node `_tN = name(args)`; returns its index.
    bool parse_call(cardinal::string name, isize& out_idx) {
        if (!expect(Tok::LParen, "'('")) return false;
        cardinal::vector<Arg> args;
        if (cur_.kind != Tok::RParen) {
            for (;;) {
                Primary a;
                if (!parse_primary(a)) return false;
                args.push_back(a.arg);
                if (cur_.kind == Tok::Comma) { advance(); continue; }
                break;
            }
        }
        if (!expect(Tok::RParen, "')'")) return false;
        Node n;
        n.kind = NodeKind::Call;
        n.out  = temp_name();
        n.fn   = cardinal::move(name);
        n.args = cardinal::move(args);
        out_idx = static_cast<isize>(fn_->nodes.size());
        n.id   = next_id_++;
        fn_->nodes.push_back(cardinal::move(n));
        return true;
    }

    bool parse_statement() {
        if (cur_.kind == Tok::KwLet) {
            advance();
            if (cur_.kind != Tok::Ident) return fail("expected name after 'let'");
            cardinal::string name = cur_.text;
            advance();
            if (!expect(Tok::Equals, "'='")) return false;
            Primary p;
            if (!parse_primary(p)) return false;
            if (p.node_idx >= 0) {
                // RHS was a call: name that node directly (no alias node).
                fn_->nodes[static_cast<usize>(p.node_idx)].out = cardinal::move(name);
            } else if (p.arg.is_literal) {
                Node n; n.kind = NodeKind::Literal; n.out = cardinal::move(name);
                n.literal = p.arg.literal; n.id = next_id_++;
                fn_->nodes.push_back(cardinal::move(n));
            } else {
                // `let x = y` — a reroute/copy node.
                Node n; n.kind = NodeKind::Call; n.out = cardinal::move(name);
                n.fn = "copy"; n.args.push_back(p.arg); n.id = next_id_++;
                fn_->nodes.push_back(cardinal::move(n));
            }
            return true;
        }
        if (cur_.kind == Tok::KwReturn) {
            advance();
            Node n; n.kind = NodeKind::Return; n.id = next_id_++;
            if (cur_.kind != Tok::RBrace && cur_.kind != Tok::End) {
                Primary p;
                if (!parse_primary(p)) return false;
                n.args.push_back(p.arg);
            }
            fn_->nodes.push_back(cardinal::move(n));
            return true;
        }
        // Expression statement: must be a call (an action / exec node).
        if (cur_.kind == Tok::Ident) {
            cardinal::string name = cur_.text;
            advance();
            isize idx = -1;
            if (!parse_call(cardinal::move(name), idx)) return false;
            fn_->nodes[static_cast<usize>(idx)].out.clear();   // exec node, no output
            return true;
        }
        return fail("expected 'let', 'return', or a call");
    }

    bool parse_func(Function& f) {
        advance();   // consume 'func'
        if (cur_.kind != Tok::Ident) return fail("expected function name");
        f.name = cur_.text;
        advance();
        if (!expect(Tok::LParen, "'('")) return false;
        if (cur_.kind != Tok::RParen) {
            for (;;) {
                if (cur_.kind != Tok::Ident) return fail("expected parameter name");
                Param p; p.name = cur_.text; advance();
                if (cur_.kind == Tok::Colon) {
                    advance();
                    if (cur_.kind != Tok::Ident) return fail("expected type after ':'");
                    p.type = cur_.text; advance();
                }
                f.params.push_back(cardinal::move(p));
                if (cur_.kind == Tok::Comma) { advance(); continue; }
                break;
            }
        }
        if (!expect(Tok::RParen, "')'")) return false;
        if (!expect(Tok::LBrace, "'{'")) return false;

        fn_      = &f;
        next_id_ = 0;
        temp_n_  = 0;
        while (cur_.kind != Tok::RBrace) {
            if (cur_.kind == Tok::End)   return fail("unexpected end of input (missing '}')");
            if (cur_.kind == Tok::Error) return fail(cur_.text.c_str());
            if (!parse_statement()) return false;
        }
        advance();   // consume '}'
        fn_ = nullptr;
        return true;
    }

    // --- helpers ------------------------------------------------------------
    void advance() { cur_ = lex_.next(); }
    bool expect(Tok k, const char* what) {
        if (cur_.kind != k) return fail(what, "expected ");
        advance();
        return true;
    }
    cardinal::string temp_name() {
        char buf[24];
        cardinal::snprintf(buf, sizeof(buf), "_t%u", temp_n_++);
        return cardinal::string(buf);
    }
    bool fail(const char* msg, const char* prefix = "") {
        if (err_ != nullptr) {
            char buf[160];
            cardinal::snprintf(buf, sizeof(buf), "line %u: %s%s",
                               cur_.line, prefix, msg);
            *err_ = buf;
        }
        return false;
    }

    Lexer            lex_;
    Token            cur_;
    cardinal::string* err_{nullptr};
    Function*        fn_{nullptr};
    u32              next_id_{0};
    u32              temp_n_{0};
};

}  // namespace

bool parse_source(const cardinal::string& src, Graph& out, cardinal::string* err) {
    out.funcs.clear();
    Parser p(src, err);
    return p.parse(out);
}

bool canonicalize(const cardinal::string& src, cardinal::string& out,
                  cardinal::string* err) {
    Graph g;
    if (!parse_source(src, g, err)) return false;
    out = generate_source(g);
    return true;
}

}  // namespace cardinal::blueprint
