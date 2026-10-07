// include/strata/glm/json.hpp - the smallest JSON reader the GLM engine needs.
//
// Two inputs are JSON: config.json (and generation_config.json) and the header of every safetensors shard.  Both
// come from a model folder the user downloaded, so the parser refuses rather than guesses: a malformed document is
// an error with an offset, never a partial tree.  Numbers are kept as double (the header's byte offsets reach
// 3.1e9, well inside double's exact integer range of 2^53).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::glm {

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    bool is_null() const { return type == Type::Null; }
    bool is_num() const { return type == Type::Number; }
    bool is_str() const { return type == Type::String; }
    bool is_arr() const { return type == Type::Array; }
    bool is_obj() const { return type == Type::Object; }

    /// The member `k`, or nullptr when this is not an object or has no such key.
    const Json* get(const std::string& k) const {
        if (type != Type::Object) return nullptr;
        auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    }
    int64_t as_i64(int64_t dflt = 0) const { return type == Type::Number ? (int64_t) num : dflt; }
    double as_f64(double dflt = 0.0) const { return type == Type::Number ? num : dflt; }
};

namespace detail {
struct JsonParser {
    const char* p;
    const char* end;
    std::string err;

    void ws() {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    }
    bool fail(const char* what, const char* begin) {
        if (err.empty()) err = std::string(what) + " at byte " + std::to_string(p - begin);
        return false;
    }
    static void put_utf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) s += (char) cp;
        else if (cp < 0x800) { s += (char) (0xC0 | (cp >> 6)); s += (char) (0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            s += (char) (0xE0 | (cp >> 12)); s += (char) (0x80 | ((cp >> 6) & 0x3F)); s += (char) (0x80 | (cp & 0x3F));
        } else {
            s += (char) (0xF0 | (cp >> 18)); s += (char) (0x80 | ((cp >> 12) & 0x3F));
            s += (char) (0x80 | ((cp >> 6) & 0x3F)); s += (char) (0x80 | (cp & 0x3F));
        }
    }
    bool hex4(uint32_t& v, const char* begin) {
        if (end - p < 4) return fail("short \\u escape", begin);
        v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t) (c - 'A' + 10);
            else return fail("bad \\u escape", begin);
        }
        return true;
    }
    bool string(std::string& out, const char* begin) {
        if (p >= end || *p != '"') return fail("expected a string", begin);
        ++p;
        while (p < end && *p != '"') {
            char c = *p++;
            if (c != '\\') { out += c; continue; }
            if (p >= end) return fail("unterminated escape", begin);
            char e = *p++;
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp, begin)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        p += 2;
                        uint32_t lo = 0;
                        if (!hex4(lo, begin)) return false;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: return fail("bad escape", begin);
            }
        }
        if (p >= end) return fail("unterminated string", begin);
        ++p;
        return true;
    }
    bool value(Json& v, const char* begin, int depth) {
        if (depth > 64) return fail("nesting too deep", begin);
        ws();
        if (p >= end) return fail("unexpected end", begin);
        char c = *p;
        if (c == '{') {
            ++p;
            v.type = Json::Type::Object;
            ws();
            if (p < end && *p == '}') { ++p; return true; }
            for (;;) {
                ws();
                std::string k;
                if (!string(k, begin)) return false;
                ws();
                if (p >= end || *p != ':') return fail("expected ':'", begin);
                ++p;
                Json child;
                if (!value(child, begin, depth + 1)) return false;
                v.obj[std::move(k)] = std::move(child);
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == '}') { ++p; return true; }
                return fail("expected ',' or '}'", begin);
            }
        }
        if (c == '[') {
            ++p;
            v.type = Json::Type::Array;
            ws();
            if (p < end && *p == ']') { ++p; return true; }
            for (;;) {
                Json child;
                if (!value(child, begin, depth + 1)) return false;
                v.arr.push_back(std::move(child));
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == ']') { ++p; return true; }
                return fail("expected ',' or ']'", begin);
            }
        }
        if (c == '"') { v.type = Json::Type::String; return string(v.str, begin); }
        if (end - p >= 4 && std::string(p, 4) == "true") { p += 4; v.type = Json::Type::Bool; v.b = true; return true; }
        if (end - p >= 5 && std::string(p, 5) == "false") { p += 5; v.type = Json::Type::Bool; return true; }
        if (end - p >= 4 && std::string(p, 4) == "null") { p += 4; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            const char* s = p;
            while (p < end && (std::strchr("+-0123456789.eE", *p) != nullptr)) ++p;
            std::string tok(s, p);
            char* stop = nullptr;
            v.num = std::strtod(tok.c_str(), &stop);
            if (stop == tok.c_str() || *stop != '\0') return fail("bad number", begin);
            v.type = Json::Type::Number;
            return true;
        }
        return fail("unexpected character", begin);
    }
};
}  // namespace detail

/// Parse `text` (exactly one JSON value, surrounding whitespace allowed).  On failure returns false and says where.
inline bool json_parse(const std::string& text, Json& out, std::string& err) {
    detail::JsonParser ps{text.data(), text.data() + text.size(), {}};
    out = Json{};
    if (!ps.value(out, text.data(), 0)) { err = ps.err; return false; }
    ps.ws();
    if (ps.p != ps.end) { err = "trailing bytes after the JSON value"; return false; }
    return true;
}

}  // namespace strata::glm
