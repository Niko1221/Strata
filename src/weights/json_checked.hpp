#pragma once
#include "nlohmann/json.hpp"
#include <limits>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::weights::detail {
using Json = nlohmann::json;
inline std::filesystem::path utf8_path(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}
inline void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
inline uint64_t uint64(const Json& j) {
    require(j.is_number_unsigned() || (j.is_number_integer() && j.get<int64_t>() >= 0),
            "expected unsigned integer (floats and negative integers are forbidden)");
    return j.get<uint64_t>();
}
inline uint64_t add(uint64_t a, uint64_t b) {
    require(b <= UINT64_MAX - a, "byte offset overflow");
    return a + b;
}
inline uint64_t mul(uint64_t a, uint64_t b) {
    require(a == 0 || b <= UINT64_MAX / a, "shape/byte count overflow");
    return a * b;
}
inline Json parse(const std::string& text) {
    // nlohmann 3.12's DOM callback parser scans the parent's existing members at
    // every object_end (discard cleanup). For 40,000 tensor objects this is
    // quadratic even if the callback never discards anything. Use a public SAX
    // validation pass, then the normal DOM parser; keep duplicate rejection.
    struct Validator : nlohmann::json_sax<Json> {
        std::vector<std::set<std::string>> keys;
        size_t depth = 0;
        bool null() override { return true; }
        bool boolean(bool) override { return true; }
        bool number_integer(number_integer_t) override { return true; }
        bool number_unsigned(number_unsigned_t) override { return true; }
        bool number_float(number_float_t, const string_t&) override { return true; }
        bool string(string_t&) override { return true; }
        bool binary(binary_t&) override { return true; }
        bool start_object(size_t) override {
            require(++depth <= 64, "JSON nesting exceeds 64"); keys.emplace_back(); return true;
        }
        bool key(string_t& value) override {
            require(!keys.empty() && keys.back().insert(value).second, "duplicate JSON key: " + value);
            return true;
        }
        bool end_object() override { keys.pop_back(); --depth; return true; }
        bool start_array(size_t) override { require(++depth <= 64, "JSON nesting exceeds 64"); return true; }
        bool end_array() override { --depth; return true; }
        bool parse_error(size_t, const std::string&, const nlohmann::detail::exception& ex) override {
            throw std::runtime_error(ex.what());
        }
    };
    {
        Validator validator;
        require(Json::sax_parse(text, &validator), "invalid JSON");
    }
    return Json::parse(text);
}
} // namespace strata::weights::detail
