#include "strata/program/serve_input.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
static auto parse(const std::string& data, size_t chunk) {
    size_t pos = 0;
    strata::program::ServeInput reader([&](char* out, size_t size) -> std::ptrdiff_t {
        const auto n = std::min({size, chunk, data.size() - pos});
        std::memcpy(out, data.data() + pos, n); pos += n; return (std::ptrdiff_t) n;
    });
    std::vector<strata::program::ServeRequest> requests;
    while (auto request = reader.next()) requests.push_back(std::move(*request));
    return requests;
}
int main() {
    try {
        const std::string source = "root ::= \"\\\"\" word\nword ::= \"\xe7\x8c\xab\" # STOP is data\n";
        const std::string frame = "GENG1 " + std::to_string(source.size()) + "\n" + source + "\nGEN 8 1,2,3\n";
        int cases = 0;
        for (size_t chunk : {1, 2, 7, 4096}) {
            auto valid = parse(frame + "STOP\nGEN 2 4,5\nQUIT\n", chunk);
            REQUIRE(valid.size() == 4 && valid[0].grammar == source && valid[0].line == "GEN 8 1,2,3");
            REQUIRE(!valid[0].fatal && valid[1].line == "STOP" && valid[2].grammar.empty());
            auto check = parse("GENG1 " + std::to_string(source.size()) + "\n" + source + "\nCHECKG\n" + frame, chunk);
            REQUIRE(check.size() == 2 && check[0].line == "CHECKG" && check[0].grammar == source);
            REQUIRE(!check[0].fatal && check[1].line == "GEN 8 1,2,3");
            for (const std::string flags : {"1 1", "1 0", "0 1"}) {
                const auto scoped = parse("GENG2 " + std::to_string(source.size()) + " " + flags +
                                          "\n" + source + "\nCHECKG\n" + frame, chunk);
                REQUIRE(scoped.size() == 2 && !scoped[0].fatal && scoped[0].grammar == source);
                REQUIRE(scoped[0].thinking == (flags[0] == '1') && scoped[0].tools == (flags[2] == '1'));
                REQUIRE(!scoped[1].thinking && !scoped[1].tools);
            }
            for (const std::string flags : {"1 0 0 0", "1 1 1 2048", "0 1 0 2"}) {
                auto json = parse("GENG3 " + std::to_string(source.size()) + " " + flags +
                                  "\n" + source + "\nCHECKG\n" + frame, chunk);
                REQUIRE(json.size() == 2 && !json[0].fatal && json[0].grammar == source);
                REQUIRE(json[0].json_schema == (flags[0] == '1'));
                REQUIRE(!json[1].json_schema && json[1].reasoning_tokens == 0);
            }
            for (const std::string flags : {"1 0 0 2", "1 1 0 8193", "1 1 0 -1", "2 1 1 0", "1 1 0 0 extra"}) {
                auto bad = parse("GENG3 1 " + flags + "\nx\nCHECKG\n", chunk);
                REQUIRE(bad.size() == 1 && bad[0].fatal);
            }
            for (size_t cut = 1; cut < frame.size(); ++cut) {
                auto bad = parse(frame.substr(0, cut), chunk);
                if (cut >= 4) REQUIRE(bad.size() == 1 && bad[0].fatal);
            }
            for (const std::string header : {"GENG2 1", "GENG1 -1", "GENG1 8193", "GENG1 0", "GENG1 1x",
                                            "GENG2 1 0 0", "GENG2 1 1 2", "GENG2 1 1 1 extra"}) {
                auto bad = parse(header + "\nSTOP\nGEN 2 1\n", chunk);
                REQUIRE(bad.size() == 1 && bad[0].fatal);
            }
            auto delimiter = parse("GENG1 1\naSTOP\nGEN 2 1\n", chunk);
            REQUIRE(delimiter.size() == 1 && delimiter[0].fatal);
            auto injected = parse("GENG1 1\na\nSTOP\n", chunk);
            REQUIRE(injected.size() == 1 && injected[0].fatal);
            auto nul = parse(std::string("GENG1 1\n") + '\0' + "\nGEN 2 1\n", chunk);
            REQUIRE(nul.size() == 1 && nul[0].grammar.size() == 1 && nul[0].grammar[0] == '\0');
            // NUL remains inert data for the native grammar validator to reject.
            ++cases;
        }
        REQUIRE(strata::program::protocol_error("bad\nT 42\rDONE") == "bad T 42 DONE");
        for (const std::string text : {"\xe7\x8c\xab", "\xf0\x9f\x90\x88"}) {
            for (size_t padding = 512 - text.size(); padding < 512; ++padding) {
                const std::string prefix(padding, 'x');
                REQUIRE(strata::program::protocol_error(prefix + text + "tail") ==
                        prefix + (padding + text.size() <= 512 ? text : ""));
            }
        }
        std::cout << "serve input: " << cases << " chunk sizes, all frame truncations and inert payload cases passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
