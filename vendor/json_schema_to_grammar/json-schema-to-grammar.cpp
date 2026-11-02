// Vendored verbatim from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (common/json-schema-to-grammar.cpp), MIT license (see LICENSE-llama.txt in this directory).
// Strata edits are marked with "STRATA:" and kept to the include-closure shims (json_compat.h, <cassert>)
// so the converter builds with zero llama.cpp headers.
#include "json-schema-to-grammar.h"
#include "json_compat.h"  // STRATA: shim for llama.cpp common.h — string_split only
#include "trie.h"
#include "unicode.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = common_json;

static std::string build_repetition(const std::string & item_rule, int min_items, int max_items, const std::string & separator_rule = "") {
    auto has_max = max_items != std::numeric_limits<int>::max();

    if (max_items == 0) {
        return "";
    }
    if (min_items == 0 && max_items == 1) {
        return item_rule + "?";
    }

    if (separator_rule.empty()) {
        if (min_items == 1 && !has_max) {
            return item_rule + "+";
        }
        if (min_items == 0 && !has_max) {
            return item_rule + "*";
        }
        return item_rule + "{" + std::to_string(min_items) + "," + (has_max ? std::to_string(max_items) : "") + "}";
    }

    auto result = item_rule + " " + build_repetition("(" + separator_rule + " " + item_rule + ")", min_items == 0 ? 0 : min_items - 1, has_max ? max_items - 1 : max_items);
    if (min_items == 0) {
        result = "(" + result + ")?";
    }
    return result;
}

static void build_min_max_int(int64_t min_value, int64_t max_value, std::stringstream & out, int decimals_left = 16, bool top_level = true) {
    auto has_min = min_value != std::numeric_limits<int64_t>::min();
    auto has_max = max_value != std::numeric_limits<int64_t>::max();

    auto digit_range = [&](char from, char to) {
        out << "[";
        if (from == to) {
            out << from;
        } else {
            out << from << "-" << to;
        }
        out << "]";
    };
    auto more_digits = [&](int min_digits, int max_digits) {
        out << "[0-9]";
        if (min_digits == max_digits && min_digits == 1) {
            return;
        }
        out << "{";
        out << min_digits;
        if (max_digits != min_digits) {
            out << ",";
            if (max_digits != std::numeric_limits<int>::max()) {
                out << max_digits;
            }
        }
        out << "}";
    };
    std::function<void(const std::string_view &, const std::string_view &)> uniform_range =
        [&](const std::string_view & from, const std::string_view & to) {
            size_t i = 0;
            while (i < from.length() && i < to.length() && from[i] == to[i]) {
                i++;
            }
            if (i > 0) {
                out << "\"" << from.substr(0, i) << "\"";
            }
            if (i < from.length() && i < to.length()) {
                if (i > 0) {
                    out << " ";
                }
                auto sub_len = from.length() - i - 1;
                if (sub_len > 0) {
                    auto from_sub = from.substr(i + 1);
                    auto to_sub = to.substr(i + 1);
                    auto sub_zeros = string_repeat("0", sub_len);
                    auto sub_nines = string_repeat("9", sub_len);

                    auto to_reached = false;
                    out << "(";
                    if (from_sub == sub_zeros) {
                        digit_range(from[i], to[i] - 1);
                        out << " ";
                        more_digits(sub_len, sub_len);
                    } else {
                        out << "[" << from[i] << "] ";
                        out << "(";
                        uniform_range(from_sub, sub_nines);
                        out << ")";
                        if (from[i] < to[i] - 1) {
                            out << " | ";
                            if (to_sub == sub_nines) {
                                digit_range(from[i] + 1, to[i]);
                                to_reached = true;
                            } else {
                                digit_range(from[i] + 1, to[i] - 1);
                            }
                            out << " ";
                            more_digits(sub_len, sub_len);
                        }
                    }
                    if (!to_reached) {
                        out << " | ";
                        digit_range(to[i], to[i]);
                        out << " ";
                        uniform_range(sub_zeros, to_sub);
                    }
                    out << ")";
                } else {
                    out << "[" << from[i] << "-" << to[i] << "]";
                }
            }
        };

    if (has_min && has_max) {
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_min_max_int(-max_value, -min_value, out, decimals_left, /* top_level= */ true);
            out << ")";
            return;
        }

        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(0, -min_value, out, decimals_left, /* top_level= */ true);
            out << ") | ";
            min_value = 0;
        }

        auto min_s = std::to_string(min_value);
        auto max_s = std::to_string(max_value);
        auto min_digits = min_s.length();
        auto max_digits = max_s.length();

        for (auto digits = min_digits; digits < max_digits; digits++) {
            uniform_range(min_s, string_repeat("9", digits));
            min_s = "1" + string_repeat("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
        return;
    }

    auto less_decimals = std::max(decimals_left - 1, 1);

    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(std::numeric_limits<int64_t>::min(), -min_value, out, decimals_left, /* top_level= */ false);
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else if (min_value == 0) {
            if (top_level) {
                out << "[0] | [1-9] ";
                more_digits(0, less_decimals);
            } else {
                more_digits(1, decimals_left);
            }
        } else if (min_value <= 9) {
            char c = '0' + min_value;
            auto range_start = top_level ? '1' : '0';
            if (c > range_start) {
                digit_range(range_start, c - 1);
                out << " ";
                more_digits(1, less_decimals);
                out << " | ";
            }
            digit_range(c, '9');
            out << " ";
            more_digits(0, less_decimals);
        } else {
            auto min_s = std::to_string(min_value);
            auto len = min_s.length();
            auto c = min_s[0];

            if (c > '1') {
                digit_range(top_level ? '1' : '0', c - 1);
                out << " ";
                more_digits(len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_max_int(std::stoll(min_s.substr(1)), std::numeric_limits<int64_t>::max(), out, less_decimals, /* top_level= */ false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range(c + 1, '9');
                out << " ";
                more_digits(len - 1, less_decimals);
            }
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                out << "\"-\" [1-9] ";
                more_digits(0, less_decimals);
                out << " | ";
            }
            build_min_max_int(0, max_value, out, decimals_left, /* top_level= */ true);
        } else {
            out << "\"-\" (";
            build_min_max_int(-max_value, std::numeric_limits<int64_t>::max(), out, decimals_left, /* top_level= */ false);
            out << ")";
        }
        return;
    }

    throw std::runtime_error("At least one of min_value or max_value must be set");
}

const std::string SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};

static std::unordered_map<std::string, BuiltinRule> PRIMITIVE_RULES = {
    {"boolean", {"(\"true\" | \"false\")", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part)", {"integral-part"}}},
    {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
    {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", {"string", "value"}}},
    {"array", {"\"[\" space ( value (\",\" space value)* )? space \"]\"", {"value"}}},
    {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"", {}}},
    {"char",   {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
    {"string", {"\"\\\"\" char* \"\\\"\"", {"char"}}},
    {"null", {"\"null\"", {}}},
};

static std::unordered_map<std::string, BuiltinRule> STRING_FORMAT_RULES = {
    {"uuid-raw", {"[0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12}", {}}},
    {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
    {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
    {"date-time", {"date \"T\" time", {"date", "time"}}},
    {"date-string", {"\"\\\"\" date \"\\\"\"", {"date"}}},
    {"time-string", {"\"\\\"\" time \"\\\"\"", {"time"}}},
    {"date-time-string", {"\"\\\"\" date-time \"\\\"\"", {"date-time"}}}
};

static bool is_reserved_name(const std::string & name) {
    static const std::unordered_set<std::string> RESERVED_NAMES = [] {
        std::unordered_set<std::string> s;
        s.insert("root");
        for (const auto & p : PRIMITIVE_RULES) {
            s.insert(p.first);
        }
        for (const auto & p : STRING_FORMAT_RULES) {
            s.insert(p.first);
        }
        return s;
    }();
    return RESERVED_NAMES.find(name) != RESERVED_NAMES.end();
}

static std::regex INVALID_RULE_CHARS_RE("[^a-zA-Z0-9-]+");
static std::regex GRAMMAR_LITERAL_ESCAPE_RE("[\r\n\"\\\\]");
static std::regex GRAMMAR_RANGE_LITERAL_ESCAPE_RE("[\r\n\"\\]\\-\\\\]");
static std::unordered_map<char, std::string> GRAMMAR_LITERAL_ESCAPES = {
    {'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""}, {'-', "\\-"}, {']', "\\]"}, {'\\', "\\\\"}
};

static const int MAX_PATTERN_DEPTH = 100;

static std::unordered_set<char> NON_LITERAL_SET = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?', '^', '$'};
static std::unordered_set<char> ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = {'^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?'};

// \d \w \s (and their negations) stand for a fixed set of code points, so they are expanded into character
// classes before the pattern is tokenized instead of being reported as unsupported - client schemas spell
// `^\d+$` and `^[\w.-]+$` constantly.  Inside a class only the positive forms expand (a union of two
// complements is not something a GBNF class can say); the negated forms become negated classes outside one,
// and '<' is excluded from those by the same code that handles a written-out class.
static std::string expand_shorthand_classes(const std::string & pattern) {
    constexpr const char * digits = "0-9";
    constexpr const char * word = "0-9A-Za-z_";
    constexpr const char * space = " \\t\\n\\r\\x0B\\x0C";
    std::string out;
    out.reserve(pattern.size());
    bool in_class = false;
    bool at_class_start = false;  // a ']' here is a member, not the end of the class
    for (size_t i = 0; i < pattern.size();) {
        const char c = pattern[i];
        if (c == '\\' && i + 1 < pattern.size()) {
            const char escape = pattern[i + 1];
            const char * positive = escape == 'd' ? digits : escape == 'w' ? word : escape == 's' ? space : nullptr;
            const char * negative = escape == 'D' ? digits : escape == 'W' ? word : escape == 'S' ? space : nullptr;
            if (positive) {
                out += in_class ? positive : std::string("[") + positive + "]";
            } else if (negative && !in_class) {
                out += std::string("[^") + negative + "]";
            } else {
                out += pattern.substr(i, 2);  // any other escape is left to the tokenizer to judge
            }
            i += 2;
            at_class_start = false;
            continue;
        }
        if (!in_class) {
            if (c == '[') {
                in_class = true;
                at_class_start = true;
            }
        } else if (at_class_start) {
            at_class_start = (c == '^');  // only a negation keeps the class at its start
        } else if (c == ']') {
            in_class = false;
        }
        out += c;
        ++i;
    }
    return out;
}

static std::string replacePattern(const std::string & input, const std::regex & regex, const std::function<std::string(const std::smatch  &)> & replacement) {
    std::smatch match;
    std::string result;

    std::string::const_iterator searchStart(input.cbegin());
    std::string::const_iterator searchEnd(input.cend());

    while (std::regex_search(searchStart, searchEnd, match, regex)) {
        result.append(searchStart, searchStart + match.position());
        result.append(replacement(match));
        searchStart = match.suffix().first;
    }

    result.append(searchStart, searchEnd);

    return result;
}

static std::string format_literal(const std::string & literal) {
    std::string escaped = replacePattern(literal, GRAMMAR_LITERAL_ESCAPE_RE, [&](const std::smatch & match) {
        char c = match.str()[0];
        return GRAMMAR_LITERAL_ESCAPES.at(c);
    });
    return "\"" + escaped + "\"";
}

std::string gbnf_format_literal(const std::string & literal) { return format_literal(literal); }

// JSON strings embedded in a tool parameter are later parsed by the chat-template
// parser.  Keep protocol marker characters out of the serialized JSON text while
// preserving their decoded value.  The converter's ordinary response-format path
// must retain its historical output, so this is selected only by tool options.
static std::string format_json_literal(const std::string & literal, bool protect_angle) {
    if (!protect_angle) {
        return format_literal(literal);
    }
    std::string escaped;
    escaped.reserve(literal.size());
    for (char c : literal) {
        if (c == '<') {
            escaped += "\\u003c";
        } else {
            escaped.push_back(c);
        }
    }
    return format_literal(escaped);
}

static size_t gbnf_escape_length(const std::string & pattern, size_t pos) {
    if (pos + 1 >= pattern.length() || pattern[pos] != '\\') {
        return 0;
    }
    size_t n_hex = 0;
    switch (pattern[pos + 1]) {
        case 'x': n_hex = 2; break;
        case 'u': n_hex = 4; break;
        case 'U': n_hex = 8; break;
        // keep in sync with parse_char() in src/llama-grammar.cpp
        case 't': case 'r': case 'n': case '\\': case '"': case '[': case ']': case '-':
            return 2;
        default:
            return 0;
    }
    if (pos + 2 + n_hex > pattern.length()) {
        return 0;
    }
    for (size_t i = pos + 2; i < pos + 2 + n_hex; i++) {
        char h = pattern[i];
        if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'))) {
            return 0;
        }
    }
    return 2 + n_hex;
}

class common_chat_schema_converter {
private:
    friend std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options);
    bool _dotall;
    bool _tool_value;
    std::string _forbidden_sequence;
    std::map<std::string, std::string> _rules;
    std::unordered_set<std::string> _refs_being_resolved;
    std::vector<std::string> _errors;
    std::vector<std::string> _warnings;

    template <typename T>
    static const T & as(const common_chat_schema & node) {
        return static_cast<const T &>(node);
    }

    std::string _add_rule(const std::string & name, const std::string & rule) {
        std::string esc_name = regex_replace(name, INVALID_RULE_CHARS_RE, "-");
        if (_rules.find(esc_name) == _rules.end() || _rules[esc_name] == rule) {
            _rules[esc_name] = rule;
            return esc_name;
        }
        int i = 0;
        while (_rules.find(esc_name + std::to_string(i)) != _rules.end() && _rules[esc_name + std::to_string(i)] != rule) {
            i++;
        }
        std::string key = esc_name + std::to_string(i);
        _rules[key] = rule;
        return key;
    }

    std::string _generate_union_rule(const std::string & name, const std::vector<common_chat_schema_ptr> & alt_schemas) {
        std::vector<std::string> rules;
        rules.reserve(alt_schemas.size());
        for (size_t i = 0; i < alt_schemas.size(); i++) {
            rules.push_back(visit(*alt_schemas[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        }
        return string_join(rules, " | ");
    }

    // thrown when the pattern is a valid regex with no grammar equivalent
    struct unsupported_pattern : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // thrown when the pattern is not a valid regex
    struct invalid_pattern : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // One code point of a GBNF character class or string literal, decoded; `i` advances past its raw
    // spelling.  Decoding matters because '<' can be spelled as "\x3c" / "\u003c" / "\U0000003c" as well
    // as directly, and the raw spelling is what the compiled rule actually matches.
    static uint32_t _gbnf_codepoint(const std::string & text, size_t & i) {
        if (text[i] != '\\' || i + 1 >= text.size()) {
            const utf8_parse_result r = common_parse_utf8_codepoint(text, i);
            if (r.status == utf8_parse_result::SUCCESS) {
                i += r.bytes_consumed;
                return r.codepoint;
            }
            return (unsigned char) text[i++];
        }
        const char escape = text[i + 1];
        const size_t hex = escape == 'x' ? 2 : escape == 'u' ? 4 : escape == 'U' ? 8 : 0;
        if (hex) {
            const std::string digits = text.substr(i + 2, hex);
            i += 2 + hex;
            return (uint32_t) std::strtoul(digits.c_str(), nullptr, 16);
        }
        i += 2;
        switch (escape) {
            case 'n': return '\n';
            case 'r': return '\r';
            case 't': return '\t';
            case '0': return 0;
            default:  return (unsigned char) escape;
        }
    }

    // A pattern compiles to the code points it matches, so it can match '<' - which no protected value may
    // contain, because the parameter closer starts with one.  Excluding '<' from the character classes
    // NARROWS the language the grammar accepts, which is the safe direction (a value can come out outside
    // the pattern, never a call cut short); a pattern that REQUIRES '<' has nothing left to accept, so the
    // caller reports it instead of emitting a rule that can never match.
    static std::string _exclude_lt_in_class(const std::string & inner, bool negated, bool & requires_lt) {
        constexpr uint32_t LT = '<';
        std::vector<std::string> pieces;
        bool excludes_lt = false;
        size_t k = 0;
        // one class member, returned exactly as written
        auto read_member = [&](uint32_t & value) {
            const size_t start = k;
            value = _gbnf_codepoint(inner, k);
            return inner.substr(start, k - start);
        };
        while (k < inner.size()) {
            uint32_t lo = 0;
            const std::string lo_tok = read_member(lo);
            if (k + 1 < inner.size() && inner[k] == '-') {
                k++;  // the dash
                uint32_t hi = 0;
                const std::string hi_tok = read_member(hi);
                if (lo > LT || LT > hi) {
                    pieces.push_back(lo_tok + "-" + hi_tok);
                } else if (negated) {
                    // `[^ -?]` already refuses '<', and dropping the range would WIDEN the class
                    excludes_lt = true;
                    pieces.push_back(lo_tok + "-" + hi_tok);
                } else {
                    // keep what is left on either side of '<' (0x3B is ';', 0x3D is '=')
                    excludes_lt = true;
                    if (lo < LT) pieces.push_back(lo_tok + "-\\x3b");
                    if (LT < hi) pieces.push_back("\\x3d-" + hi_tok);
                }
                continue;
            }
            if (lo == LT) {
                excludes_lt = true;
                if (negated) pieces.push_back(lo_tok);  // a negated class listing '<' is what we want
            } else {
                pieces.push_back(lo_tok);
            }
        }
        if (!negated && pieces.empty()) {
            requires_lt = true;
            return "";
        }
        std::string out;
        if (negated && !excludes_lt) out += "\\x3c";
        for (const auto & piece : pieces) out += piece;
        return out;
    }

    void _exclude_lt_in_body(std::string & body, bool & requires_lt) const {
        std::string out;
        size_t i = 0;
        while (i < body.size()) {
            if (body[i] == '"') {
                const size_t start = i++;
                while (i < body.size() && body[i] != '"') {
                    if (body[i] == '\\' && i + 1 < body.size()) i++;
                    i++;
                }
                if (i < body.size()) i++;
                const std::string literal = body.substr(start, i - start);
                if (!_forbidden_sequence.empty() && literal.find(_forbidden_sequence) != std::string::npos) {
                    throw unsupported_pattern("the pattern spells the parameter closer");
                }
                // a literal can spell '<' through an escape too, and a raw '<' handed to the model is what
                // the closer needs
                size_t k = 0;
                while (k < literal.size()) {
                    if (_gbnf_codepoint(literal, k) == '<') {
                        requires_lt = true;
                        break;
                    }
                }
                out += literal;
                continue;
            }
            if (body[i] == '[') {
                size_t j = i + 1;
                const bool negated = j < body.size() && body[j] == '^';
                if (negated) j++;
                const size_t inner_start = j;
                while (j < body.size() && body[j] != ']') {
                    if (body[j] == '\\' && j + 1 < body.size()) j++;
                    j++;
                }
                if (j >= body.size()) {  // unterminated: leave the rest untouched
                    out += body.substr(i);
                    break;
                }
                out += "[";
                if (negated) out += "^";
                out += _exclude_lt_in_class(body.substr(inner_start, j - inner_start), negated, requires_lt);
                out += "]";
                i = j + 1;
                continue;
            }
            out += body[i++];
        }
        body = out;
    }

    // every rule the call added or changed, with '<' excluded from its character classes
    void _exclude_lt_in_new_rules(const std::map<std::string, std::string> & snapshot) {
        if (_forbidden_sequence.empty()) return;
        bool requires_lt = false;
        for (auto & [rule_name, body] : _rules) {
            const auto it = snapshot.find(rule_name);
            if (it != snapshot.end() && it->second == body) continue;
            _exclude_lt_in_body(body, requires_lt);
        }
        if (requires_lt) {
            throw unsupported_pattern("the pattern needs a raw '<', which a protected value cannot contain");
        }
    }

    std::string _visit_pattern(const std::string & pattern, const std::string & name, bool raw = false) {
        auto rules_snapshot = _rules;
        try {
            const std::string rule = _pattern_to_rule(pattern, name, raw);
            _exclude_lt_in_new_rules(rules_snapshot);
            return rule;
        } catch (const unsupported_pattern & err) {
            // Revert rules. In strict tool mode, widening an unsupported pattern to any string would
            // silently defeat the schema, so report an error instead of accepting a weaker grammar.
            _rules = std::move(rules_snapshot);
            if (!_forbidden_sequence.empty()) {
                _errors.push_back("Pattern " + pattern + " is not supported for a protected tool value (" + err.what() + ")");
                return "";
            }
            _warnings.push_back("pattern " + pattern + " is not supported (" + err.what() + "), accepting any string");
            return _add_rule(name, _add_primitive("string", PRIMITIVE_RULES.at("string")));
        } catch (const invalid_pattern & err) {
            _rules = std::move(rules_snapshot);
            _errors.push_back("Invalid pattern " + pattern + ": " + err.what());
            return "";
        }
    }

    std::string _pattern_to_rule(const std::string & pattern, const std::string & name, bool raw = false) {
        if (pattern.length() < 2 || pattern.front() != '^' || pattern.back() != '$') {
            throw unsupported_pattern("not anchored with '^' and '$'");
        }
        // the shorthand expansion is part of the strict tool-value path only: on the ordinary path an
        // unsupported pattern keeps llama.cpp's warn-and-widen behaviour, so these goldens stay identical
        std::string sub_pattern = _tool_value ? expand_shorthand_classes(pattern.substr(1, pattern.length() - 2))
                                              : pattern.substr(1, pattern.length() - 2);
        std::unordered_map<std::string, std::string> sub_rule_ids;

        size_t i = 0;
        size_t length = sub_pattern.length();
        int paren_depth = 0;

        using literal_or_rule = std::pair<std::string, bool>;
        auto to_rule = [&](const literal_or_rule & ls) {
            auto is_literal = ls.second;
            auto s = ls.first;
            return is_literal ? "\"" + s + "\"" : s;
        };
        std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
            std::vector<literal_or_rule> seq;

            auto get_dot = [&]() {
                std::string rule;
                if (_dotall) {
                    rule = "[\\U00000000-\\U0010FFFF]";
                } else {
                    rule = "[^\\x0A\\x0D]";
                }
                return _add_rule("dot", rule);
            };

            // Joins the sequence, merging consecutive literals together.
            auto join_seq = [&]() {
                std::vector<literal_or_rule> ret;

                std::string literal;
                auto flush_literal = [&]() {
                    if (literal.empty()) {
                        return false;
                    }
                    ret.emplace_back(literal, true);
                    literal.clear();
                    return true;
                };

                for (const auto & item : seq) {
                    auto is_literal = item.second;
                    if (is_literal) {
                        literal += item.first;
                    } else {
                        flush_literal();
                        ret.push_back(item);
                    }
                }
                flush_literal();

                std::vector<std::string> results;
                results.reserve(ret.size());
                for (const auto & item : ret) {
                    results.push_back(to_rule(item));
                }
                return std::make_pair(string_join(results, " "), false);
            };

            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.emplace_back(get_dot(), false);
                    i++;
                } else if (c == '(') {
                    i++;
                    if (i < length && sub_pattern[i] == '?') {
                        if (i + 1 < length && sub_pattern[i + 1] == ':') {
                            i += 2; // skip "?:" for non-capturing group, treat as regular group
                        } else {
                            // lookaround, named group, inline flags, ...
                            throw unsupported_pattern("unsupported group syntax");
                        }
                    }
                    paren_depth++;
                    if (paren_depth > MAX_PATTERN_DEPTH) {
                        throw unsupported_pattern("pattern nesting too deep");
                    }
                    seq.emplace_back("(" + to_rule(transform()) + ")", false);
                } else if (c == ')') {
                    i++;
                    if (paren_depth == 0) {
                        throw invalid_pattern("unbalanced parentheses");
                    }
                    paren_depth--;
                    return join_seq();
                } else if (c == '^' || c == '$') {
                    throw unsupported_pattern("anchor inside the pattern");
                } else if (c == '[') {
                    std::string square_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != ']') {
                        if (sub_pattern[i] == '\\') {
                            auto escape_length = gbnf_escape_length(sub_pattern, i);
                            if (escape_length == 0) {
                                throw unsupported_pattern("unsupported escape in character class: " + sub_pattern.substr(i, 2));
                            }
                            square_brackets += sub_pattern.substr(i, escape_length);
                            i += escape_length;
                        } else {
                            square_brackets += sub_pattern[i];
                            i++;
                        }
                    }
                    if (i >= length) {
                        throw invalid_pattern("unterminated character class");
                    }
                    square_brackets += ']';
                    i++;
                    seq.emplace_back(square_brackets, false);
                } else if (c == '|') {
                    seq.emplace_back("|", false);
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    if (seq.empty()) {
                        throw invalid_pattern("nothing to repeat");
                    }
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    std::string curly_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') {
                        curly_brackets += sub_pattern[i];
                        i++;
                    }
                    if (i >= length) {
                        throw unsupported_pattern("unterminated curly brackets");
                    }
                    curly_brackets += '}';
                    i++;
                    auto nums = string_split(curly_brackets.substr(1, curly_brackets.length() - 2), ",");
                    int min_times = 0;
                    int max_times = std::numeric_limits<int>::max();
                    if (nums.size() != 1 && nums.size() != 2) {
                        throw unsupported_pattern("wrong number of values in curly brackets");
                    }
                    try {
                        if (nums.size() == 1) {
                            min_times = max_times = std::stoi(nums[0]);
                        } else {
                            if (!nums[0].empty()) {
                                min_times = std::stoi(nums[0]);
                            }
                            if (!nums[1].empty()) {
                                max_times = std::stoi(nums[1]);
                            }
                        }
                    } catch (const std::logic_error &) {
                        throw unsupported_pattern("invalid number in curly brackets");
                    }
                    if (seq.empty()) {
                        throw invalid_pattern("nothing to repeat");
                    }
                    auto &last = seq.back();
                    auto &sub = last.first;
                    auto sub_is_literal = last.second;

                    if (!sub_is_literal) {
                        std::string & sub_id = sub_rule_ids[sub];
                        if (sub_id.empty()) {
                            sub_id = _add_rule(name + "-" + std::to_string(sub_rule_ids.size()), sub);
                        }
                        sub = sub_id;
                    }
                    seq.back().first = build_repetition(
                        sub_is_literal ? "\"" + sub + "\"" : sub,
                        min_times,
                        max_times,
                        ""
                    );
                    seq.back().second = false;
                } else {
                    std::string literal;
                    auto is_non_literal = [&](char c) {
                        return NON_LITERAL_SET.find(c) != NON_LITERAL_SET.end();
                    };
                    while (i < length) {
                        if (sub_pattern[i] == '\\') {
                            if (i == length - 1) {
                                throw invalid_pattern("trailing backslash");
                            }
                            char next = sub_pattern[i + 1];
                            if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.find(next) != ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.end()) {
                                i++;
                                literal += sub_pattern[i];
                                i++;
                            } else {
                                auto escape_length = gbnf_escape_length(sub_pattern, i);
                                if (escape_length == 0) {
                                    throw unsupported_pattern("unsupported escape: " + sub_pattern.substr(i, 2));
                                }
                                literal += sub_pattern.substr(i, escape_length);
                                i += escape_length;
                            }
                        } else if (sub_pattern[i] == '"') {
                            literal += "\\\"";
                            i++;
                        } else if (!is_non_literal(sub_pattern[i]) &&
                                (i == length - 1 || literal.empty() || sub_pattern[i + 1] == '.' || !is_non_literal(sub_pattern[i + 1]))) {
                            literal += sub_pattern[i];
                            i++;
                        } else {
                            break;
                        }
                    }
                    if (literal.empty()) { // nothing was consumed, ex. a stray ']' or '}'
                        throw unsupported_pattern(std::string("unsupported character: ") + c);
                    }
                    seq.emplace_back(literal, true);
                }
            }
            return join_seq();
        };

        auto rule = to_rule(transform());
        if (paren_depth != 0) {
            throw invalid_pattern("unbalanced parentheses");
        }

        return raw ? _add_rule(name, "(" + rule + ")")
                   : _add_rule(name, "\"\\\"\" (" + rule + ") \"\\\"\"");
    }

    /*
        Returns a rule that matches a JSON string that is none of the provided strings

        not_strings({"a"})
            -> ["] ( [a] char+ | [^"a] char* )? ["]
        not_strings({"and", "also"})
            -> ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]
    */
    std::string _not_strings(const std::vector<std::string> & strings) {
        common_trie trie(strings);

        std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
        // The fallback class below is a literal character class, so it does not go through `char` and would
        // admit a raw '<' on its own.  Under a protected tool value the closer must stay unspellable, so '<'
        // joins every node's reject set.
        const std::string extra_reject = _forbidden_sequence.empty() ? std::string() : std::string("<");
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(size_t)> visit = [&](size_t idx) {
            const auto & node = trie.nodes[idx];
            std::string rejects = extra_reject;
            auto first = true;
            for (const auto & [cpt, child] : node.children) {
                std::string c = common_unicode_cpt_to_utf8(cpt);
                rejects += c;
                if (first) {
                    first = false;
                } else {
                    out << " | ";
                }
                out << "[" << c << "]";
                if (!trie.nodes[child].children.empty()) {
                    out << " (";
                    visit(child);
                    out << ")";
                } else {
                    out << " " << char_rule << "+";
                }
            }
            if (!node.children.empty()) {
                out << " | [^\"" << rejects << "] " << char_rule << "*";
            }
        };
        visit(0);

        out << " )";
        if (trie.nodes[0].pattern < 0) {
            out << "?";
        }
        out << " [\"]";
        return out.str();
    }

    std::string _resolve_ref(const common_chat_schema_ref & schema) {
        auto it = schema.ref.find('#');
        std::string ref_fragment = it != std::string::npos ? schema.ref.substr(it + 1) : schema.ref;
        static const std::regex nonalphanumeric_regex(R"([^a-zA-Z0-9-]+)");
        std::string ref_name = "ref" + std::regex_replace(ref_fragment, nonalphanumeric_regex, "-");
        if (_rules.find(ref_name) == _rules.end() && _refs_being_resolved.find(schema.ref) == _refs_being_resolved.end()) {
            if (!schema.target) {
                _errors.push_back("Unresolved $ref " + schema.ref);
                return "";
            }
            _refs_being_resolved.insert(schema.ref);
            ref_name = visit(*schema.target, ref_name);
            _refs_being_resolved.erase(schema.ref);
        }
        return ref_name;
    }

    std::string _build_object_rule(
        const std::vector<std::pair<std::string, const common_chat_schema *>> & properties,
        const std::unordered_set<std::string> & required,
        const std::string & name,
        const common_chat_schema * additional_properties)
    {
        std::vector<std::string> required_props;
        std::vector<std::string> optional_props;
        std::unordered_map<std::string, std::string> prop_kv_rule_names;
        std::vector<std::string> prop_names;
        for (const auto & kv : properties) {
            const auto &prop_name = kv.first;
            const auto &prop_schema = kv.second;
            std::string prop_rule_name = visit(*prop_schema, name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = _add_rule(
                name + (name.empty() ? "" : "-") + prop_name + "-kv",
                format_json_literal(json(prop_name).dump(), !_forbidden_sequence.empty()) + " space \":\" space " + prop_rule_name
            );
            if (required.find(prop_name) != required.end()) {
                required_props.push_back(prop_name);
            } else {
                optional_props.push_back(prop_name);
            }
            prop_names.push_back(prop_name);
        }
        if (additional_properties) {
            std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            std::string value_rule =
                additional_properties->kind() != common_chat_schema::KIND_ANY ? visit(*additional_properties, sub_name + "-value")
                : _add_primitive("value", PRIMITIVE_RULES.at("value"));

            auto key_rule =
                prop_names.empty() ? _add_primitive("string", PRIMITIVE_RULES.at("string"))
                : _add_rule(sub_name + "-k", _not_strings(prop_names));
            std::string kv_rule = _add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        if (required_props.empty() && optional_props.empty()) {
            return "\"{\" space \"}\"";
        }

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) {
                rule += " \",\" space ";
            }
            rule += prop_kv_rule_names[required_props[i]];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) {
                rule += " \",\" space ( ";
            }

            std::function<std::string(const std::vector<std::string> &, bool)> get_recursive_refs = [&](const std::vector<std::string> & ks, bool first_is_optional) {
                std::string res;
                if (ks.empty()) {
                    return res;
                }
                const std::string& k = ks[0];
                std::string kv_rule_name = prop_kv_rule_names[k];
                std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
                if (first_is_optional) {
                    res = comma_ref + (k == "*" ? "*" : "?");
                } else {
                    res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
                }
                if (ks.size() > 1) {
                    res += " " + _add_rule(
                        name + (name.empty() ? "" : "-") + k + "-rest",
                        get_recursive_refs(std::vector<std::string>(ks.begin() + 1, ks.end()), true)
                    );
                }
                return res;
            };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) {
                    rule += " | ";
                }
                rule += get_recursive_refs(std::vector<std::string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) {
                rule += " )";
            }
            rule += " )?";
        }

        rule += " space \"}\"";

        return rule;
    }

    static std::string _class_escape(char c) {
        switch (c) {
            case '\\': return "\\\\";
            case ']': return "\\]";
            case '-': return "\\-";
            case '\\n': return "\\n";
            case '\\r': return "\\r";
            case '\\t': return "\\t";
            default: return std::string(1, c);
        }
    }

    int _forbidden_next(size_t state, char ch) const {
        std::string candidate = _forbidden_sequence.substr(0, state);
        candidate.push_back(ch);
        if (candidate.size() >= _forbidden_sequence.size() &&
            candidate.compare(candidate.size() - _forbidden_sequence.size(), _forbidden_sequence.size(), _forbidden_sequence) == 0) {
            return -1;
        }
        const size_t max_size = std::min(_forbidden_sequence.size() - 1, candidate.size());
        for (size_t size = max_size; size > 0; --size) {
            if (candidate.compare(candidate.size() - size, size, _forbidden_sequence, 0, size) == 0) {
                return (int) size;
            }
        }
        return 0;
    }

    // A JSON escape branch that spells exactly one decoded character per branch: the simple escapes do,
    // and "\\uXXXX" does unless it opens a surrogate PAIR (two escapes, one character).  The pair is not
    // admitted, so a guarded string's grammar count equals the schema's character count; an astral
    // character is written as itself instead.
    static const std::string & _protected_json_escape() {
        static const std::string escaped =
            "[\\\\] ( [\"\\\\bfnrt]"
            " | \"u\" ( [0-9a-cA-Ce-fE-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]"
            "         | [dD] [0-7c-fC-F] [0-9a-fA-F] [0-9a-fA-F] ) )";
        return escaped;
    }

    std::string _guarded_string(const std::string & rule_name, int min_len, int max_len, bool raw) {
        if (_forbidden_sequence.empty()) {
            _errors.push_back("raw tool string conversion requires a forbidden parameter delimiter");
            return "";
        }
        // llama.cpp's rule for the same problem, which this mirrors: a lower bound above the threshold is
        // refused ("number of repetitions exceeds sane defaults"), an upper bound above it becomes unbounded
        // (src/llama-grammar.cpp, MAX_REPETITION_THRESHOLD).  Our engine's GBNF parser carries the same
        // threshold and the same clamp, so a schema behaves here as it does on the ordinary path.
        constexpr int MAX_GUARDED_STRING_LENGTH = 2000;
        // An upper bound the chain need not count to is unbounded in practice: the state machine exists to count
        // characters, while the property this mode guarantees (the value never spells the delimiter) holds in
        // every state.  Client schemas ask for maxLength 65535 / 1000000 to mean "a lot"; counting that far would
        // mean tens of thousands of rules for no added guarantee, so the bound is dropped - as llama.cpp does -
        // rather than the whole request refused.
        if (max_len > MAX_GUARDED_STRING_LENGTH) {
            max_len = -1;
        }
        if (min_len < 0 || max_len < -1 || min_len > MAX_GUARDED_STRING_LENGTH ||
            (max_len >= 0 && min_len > max_len)) {
            _errors.push_back("string length bounds are too large or invalid for a protected tool value (" +
                              rule_name + ": minLength " + std::to_string(min_len) + ", maxLength " +
                              std::to_string(max_len) + "; this converter counts length characters up to " +
                              std::to_string(MAX_GUARDED_STRING_LENGTH) + ")");
            return "";
        }

        const int count_limit = max_len < 0 ? min_len : max_len;
        using state_key = std::pair<size_t, int>;
        std::map<state_key, int> ids;
        std::vector<state_key> states;
        std::vector<std::string> names;
        // The state rules are written straight into _rules rather than through _add_rule, so their names are
        // sanitized here: a property name reaches rule_name with its underscores intact ("a_b"), and '_' is
        // not a GBNF word character - the whole grammar would fail to parse.
        const std::string base = regex_replace(rule_name, INVALID_RULE_CHARS_RE, "-");
        auto add_state = [&](state_key key) {
            auto it = ids.find(key);
            if (it != ids.end()) return it->second;
            const int id = (int) states.size();
            ids.emplace(key, id);
            states.push_back(key);
            std::string candidate = base + "-guard-state-" + std::to_string(id);
            while (_rules.find(candidate) != _rules.end() ||
                   std::find(names.begin(), names.end(), candidate) != names.end()) {
                candidate += "-x";
            }
            names.push_back(candidate);
            return id;
        };
        add_state({0, 0});

        // one branch per decoded character: the simple escapes spell one each, and "\uXXXX" is admitted
        // only outside the high-surrogate range, whose PAIR would spell one character with two escapes
        // and let a length-bounded string decode shorter than its grammar count
        const std::string json_escape = _protected_json_escape();
        const std::string alphabet = [&]() {
            std::string out;
            for (char c : _forbidden_sequence) {
                if (out.find(c) == std::string::npos) out.push_back(c);
            }
            return out;
        }();
        for (size_t id = 0; id < states.size(); ++id) {
            const auto [matched, count] = states[id];
            const bool can_more = max_len < 0 || count < max_len;
            const int next_count = max_len < 0 ? std::min(count + 1, count_limit) : count + 1;
            std::vector<std::string> branches;
            std::string refused;
            for (char c : alphabet) {
                if (!raw && c == '<') {
                    refused += _class_escape(c);
                    continue;
                }
                const int next = _forbidden_next(matched, c);
                if (next < 0 || next > 0 || !can_more) {
                    refused += _class_escape(c);
                }
                if (next >= 0 && next > 0 && can_more) {
                    const int next_id = add_state({(size_t) next, next_count});
                    branches.push_back(format_literal(std::string(1, c)) + " " + names[next_id]);
                }
            }
            std::string body;
            if (can_more) {
                std::string cls;
                if (raw) {
                    cls = refused.empty() ? "[\\U00000000-\\U0010FFFF]" : "[^" + refused + "]";
                } else {
                    cls = "[^\"\\\\\\x7F\\x00-\\x1F" + refused + "]";
                }
                const int free_id = add_state({0, next_count});
                body = cls + " " + names[free_id];
                if (!raw) {
                    body += " | " + json_escape + " " + names[free_id];
                }
                for (const auto & branch : branches) body += " | " + branch;
            }
            if (count >= min_len) body += (body.empty() ? "" : " | ") + std::string("\"\"");
            if (body.empty()) {
                _errors.push_back("protected string schema has no accepting production");
                body = "\"\"";
            }
            _rules[names[id]] = body;
        }
        const int start_id = ids.at({0, 0});
        const std::string framed = raw ? names[start_id] : format_literal("\"") + " " + names[start_id] + " " + format_literal("\"");
        return _add_rule(rule_name, framed);
    }

    std::string _add_primitive(const std::string & name, const BuiltinRule & rule) {
        BuiltinRule effective = rule;
        if (name == "char" && !_forbidden_sequence.empty()) {
            effective.content = "[^\"\\\\\\x7F\\x00-\\x1F<] | " + _protected_json_escape();
        }
        auto n = _add_rule(name, effective.content);
        for (const auto & dep : rule.deps) {
            BuiltinRule dep_rule;
            auto it = PRIMITIVE_RULES.find(dep);
            if (it == PRIMITIVE_RULES.end()) {
                it = STRING_FORMAT_RULES.find(dep);
                if (it == STRING_FORMAT_RULES.end()) {
                    _errors.push_back("Rule " + dep + " not known");
                    continue;
                }
            }
            if (_rules.find(dep) == _rules.end()) {
                _add_primitive(dep, it->second);
            }
        }
        return n;
    }

public:
    explicit common_chat_schema_converter(bool dotall, bool tool_value = false, std::string forbidden_sequence = {})
        : _dotall(dotall), _tool_value(tool_value), _forbidden_sequence(std::move(forbidden_sequence)) {
        _rules["space"] = SPACE_RULE;
        if (_tool_value && _forbidden_sequence.empty()) {
            _errors.push_back("tool-value conversion requires a forbidden parameter delimiter");
        }
        if (!_forbidden_sequence.empty()) {
            if (_forbidden_sequence.front() != '<') {
                _errors.push_back("protected tool delimiters must begin with '<'");
            }
            for (unsigned char c : _forbidden_sequence) {
                if (c >= 0x80) {
                    _errors.push_back("protected tool delimiters must be ASCII");
                    break;
                }
            }
        }
    }

    std::string add_schema(const std::string & name, const common_chat_schema & schema) {
        return visit(schema, name);
    }

    std::string _generate_constant_rule(const json & value, bool raw = false) {
        if (raw && value.is_string()) {
            const std::string text = value.get<std::string>();
            if (!_forbidden_sequence.empty() && text.find(_forbidden_sequence) != std::string::npos) {
                _errors.push_back("string constants containing the protected parameter closer are not supported");
            }
            return format_literal(text);
        }
        return format_json_literal(value.dump(), !_forbidden_sequence.empty());
    }

    std::string _visit_primitive(const std::string & rule_name, const std::string & type) {
        return _add_primitive(rule_name == "root" ? "root" : type, PRIMITIVE_RULES.at(type));
    }

    std::string _visit_all_of(const common_chat_schema_all_of & schema, const std::string & name, const std::string & rule_name) {
        std::unordered_set<std::string> required;
        std::vector<std::pair<std::string, const common_chat_schema *>> properties;
        std::map<std::string, size_t> enum_values;
        std::function<void(const common_chat_schema &, bool)> add_component = [&](const common_chat_schema & comp, bool is_required) {
            if (comp.kind() == common_chat_schema::KIND_REF) {
                if (const auto * target = as<common_chat_schema_ref>(comp).target) {
                    add_component(*target, is_required);
                }
            } else if (comp.kind() == common_chat_schema::KIND_OBJECT) {
                for (const auto & prop : as<common_chat_schema_object>(comp).properties) {
                    properties.emplace_back(prop.name, prop.schema.get());
                    if (is_required) {
                        required.insert(prop.name);
                    }
                }
            } else if (comp.kind() == common_chat_schema::KIND_ENUM) {
                for (const auto & v : as<common_chat_schema_enum>(comp).values) {
                    enum_values[_generate_constant_rule(v)] += 1;
                }
            }
        };
        for (const auto & child : schema.children) {
            if (child->kind() == common_chat_schema::KIND_ANY_OF) {
                for (const auto & alt : as<common_chat_schema_any_of>(*child).children) {
                    add_component(*alt, false);
                }
            } else {
                add_component(*child, true);
            }
        }
        if (!enum_values.empty()) {
            std::vector<std::string> enum_intersection;
            for (const auto & p : enum_values) {
                if (p.second == schema.children.size()) {
                    enum_intersection.push_back(p.first);
                }
            }
            if (!enum_intersection.empty()) {
                return _add_rule(rule_name, "(" + string_join(enum_intersection, " | ") + ")");
            }
        }
        return _add_rule(rule_name, _build_object_rule(properties, required, name, nullptr));
    }

    std::string visit(const common_chat_schema & schema, const std::string & name) {
        std::string rule_name = is_reserved_name(name) ? name + "-" : name.empty() ? "root" : name;
        std::string sub_name  = name + (name.empty() ? "" : "-");

        switch (schema.kind()) {
            case common_chat_schema::KIND_REF: {
                const auto & ref = as<common_chat_schema_ref>(schema);
                // --tool-value never arrives here (the strict preflight refuses $ref), and a root
                // dereference would walk a self-referential schema without a recursion guard
                return _add_rule(rule_name, _resolve_ref(ref));
            }
            case common_chat_schema::KIND_ANY_OF:
                // A union of alternatives is one production per branch, and every branch is framed the way a
                // nested value is (a string is quoted JSON, an object is its JSON spelling).  That is exactly
                // what the parameter region holds, so a nullable field - `type: ["string","null"]`, which a
                // schema generator writes for `.nullable()` - is expressible here.
                return _add_rule(rule_name, _generate_union_rule(name, as<common_chat_schema_any_of>(schema).children));
            case common_chat_schema::KIND_ALL_OF:
                if (_tool_value && name.empty()) {
                    _errors.push_back("tool parameter root allOf schemas are not supported in strict mode");
                    return "";
                }
                return _visit_all_of(as<common_chat_schema_all_of>(schema), name, rule_name);
            case common_chat_schema::KIND_CONST:
                if (_tool_value && name.empty() && as<common_chat_schema_const>(schema).value.is_string()) {
                    return _add_rule(rule_name, _generate_constant_rule(as<common_chat_schema_const>(schema).value, true));
                }
                return _add_rule(rule_name, _generate_constant_rule(as<common_chat_schema_const>(schema).value));
            case common_chat_schema::KIND_ENUM: {
                const auto & values = as<common_chat_schema_enum>(schema).values;
                bool any_string = false;
                bool all_string = !values.empty();
                for (const auto & v : values) {
                    any_string = any_string || v.is_string();
                    all_string = all_string && v.is_string();
                }
                if (_tool_value && name.empty() && any_string && !all_string) {
                    _errors.push_back("tool parameter root enum cannot mix raw strings with JSON values");
                    return "";
                }
                std::vector<std::string> enum_values;
                for (const auto & v : values) {
                    enum_values.push_back(_generate_constant_rule(v, _tool_value && name.empty() && all_string));
                }
                return _add_rule(rule_name, "(" + string_join(enum_values, " | ") + ")");
            }
            case common_chat_schema::KIND_OBJECT: {
                const auto & obj = as<common_chat_schema_object>(schema);
                if (obj.properties.empty() && obj.additional_properties && obj.additional_properties->kind() == common_chat_schema::KIND_ANY) {
                    // The `object` primitive is built from `string` and `value`, whose `char` refuses a raw
                    // '<', so a free-form object cannot spell the closer either.  This is what a schema
                    // generator writes for `record<string, unknown>`.
                    return _add_rule(rule_name, _add_primitive("object", PRIMITIVE_RULES.at("object")));
                }
                std::vector<std::pair<std::string, const common_chat_schema *>> properties;
                std::unordered_set<std::string> required;
                for (const auto & prop : obj.properties) {
                    properties.emplace_back(prop.name, prop.schema.get());
                    if (prop.required) {
                        required.insert(prop.name);
                    }
                }
                return _add_rule(rule_name, _build_object_rule(properties, required, name, obj.additional_properties.get()));
            }
            case common_chat_schema::KIND_TUPLE: {
                const auto & items = as<common_chat_schema_tuple>(schema).items;
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < items.size(); i++) {
                    if (i > 0) {
                        rule += " \",\" space ";
                    }
                    rule += visit(*items[i], sub_name + "tuple-" + std::to_string(i));
                }
                rule += " space \"]\"";
                return _add_rule(rule_name, rule);
            }
            case common_chat_schema::KIND_ARRAY: {
                const auto & arr = as<common_chat_schema_array>(schema);
                if (arr.items->kind() == common_chat_schema::KIND_ANY && arr.min_items == 0 && arr.max_items < 0) {
                    // the `array` primitive spells every element through `value`, whose strings go through
                    // `char`; a free-form array cannot spell a raw '<' either
                    return _visit_primitive(rule_name, "array");
                }
                std::string item_rule_name = visit(*arr.items, sub_name + "item");
                int max_items = arr.max_items < 0 ? std::numeric_limits<int>::max() : arr.max_items;
                return _add_rule(rule_name, "\"[\" space " + build_repetition(item_rule_name, arr.min_items, max_items, "\",\" space") + " space \"]\"");
            }
            case common_chat_schema::KIND_STRING: {
                const auto & str = as<common_chat_schema_string>(schema);
                const bool raw_root = _tool_value && name.empty();
                if (raw_root) {
                    // A root tool string is the template's raw text, so its rule is unquoted.  A pattern
                    // still applies: it is compiled to the code points it matches, with '<' excluded so no
                    // closer can be spelled (an unquoted value has no escape to spell it with).
                    if (!str.pattern.empty()) {
                        return _visit_pattern(str.pattern, rule_name, true);
                    }
                    if (str.format != common_chat_schema::FORMAT_NONE) {
                        // date/time/date-time have an unquoted rule already; uuid needs its raw spelling.
                        // Every one of them is digits and fixed punctuation, so none can produce a '<'.
                        const std::string prim_name =
                            str.format == common_chat_schema::FORMAT_UUID ? "uuid-raw"
                            : str.format == common_chat_schema::FORMAT_DATE ? "date"
                            : str.format == common_chat_schema::FORMAT_TIME ? "time"
                                                                           : "date-time";
                        return _add_rule(rule_name, _add_primitive(prim_name, STRING_FORMAT_RULES.at(prim_name)));
                    }
                    return _guarded_string(rule_name, str.min_length, str.max_length, true);
                }
                if (!str.pattern.empty()) {
                    return _visit_pattern(str.pattern, rule_name);
                }
                if (str.format == common_chat_schema::FORMAT_UUID) {
                    return _visit_primitive(rule_name, "uuid");
                }
                if (str.format != common_chat_schema::FORMAT_NONE) {
                    // date/time/uuid are spelled from digits and fixed punctuation, so none of them can
                    // produce a '<' - safe inside a protected value as well as outside one
                    std::string prim_name = std::string(str.format == common_chat_schema::FORMAT_DATE ? "date" : str.format == common_chat_schema::FORMAT_TIME ? "time" : "date-time") + "-string";
                    return _add_rule(rule_name, _add_primitive(prim_name, STRING_FORMAT_RULES.at(prim_name)));
                }
                if (!_forbidden_sequence.empty()) {
                    return _guarded_string(rule_name, str.min_length, str.max_length, false);
                }
                if (str.min_length > 0 || str.max_length >= 0) {
                    std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
                    int max_len = str.max_length < 0 ? std::numeric_limits<int>::max() : str.max_length;
                    return _add_rule(rule_name, "\"\\\"\" " + build_repetition(char_rule, str.min_length, max_len) + " \"\\\"\"");
                }
                return _visit_primitive(rule_name, "string");
            }

            case common_chat_schema::KIND_INTEGER: {
                const auto & i = as<common_chat_schema_integer>(schema);
                if (i.minimum == std::numeric_limits<int64_t>::min() && i.maximum == std::numeric_limits<int64_t>::max()) {
                    return _visit_primitive(rule_name, "integer");
                }
                std::stringstream out;
                out << "(";
                build_min_max_int(i.minimum, i.maximum, out);
                out << ")";
                return _add_rule(rule_name, out.str());
            }
            case common_chat_schema::KIND_NUMBER:
                return _visit_primitive(rule_name, "number");
            case common_chat_schema::KIND_BOOLEAN:
                return _visit_primitive(rule_name, "boolean");
            case common_chat_schema::KIND_NULL:
                return _visit_primitive(rule_name, "null");
            case common_chat_schema::KIND_ANY:
                // An unconstrained value is safe here: every string inside it goes through `char`, which
                // refuses a raw '<' and admits the JSON escape, so no production can spell the parameter
                // closer.  A schema generator writes this for a `record<string, unknown>` value, and
                // refusing it rejected tools that are perfectly expressible.
                if (_tool_value && name.empty()) {
                    // ...except at the root of a raw tool string, where the whole value IS the text
                    _errors.push_back("tool parameter roots need a concrete string, enum, const, object, array, number, boolean or null type");
                    return "";
                }
                return _add_rule(rule_name, _add_primitive("value", PRIMITIVE_RULES.at("value")));
        }
        return "";
    }

    void check_errors() {
        if (!_errors.empty()) {
            throw std::invalid_argument("JSON schema conversion failed:\n" + string_join(_errors, "\n"));
        }
        if (!_warnings.empty()) {
            fprintf(stderr, "WARNING: JSON schema conversion was incomplete: %s\n", string_join(_warnings, "; ").c_str());
        }
    }

    std::string format_grammar() {
        std::stringstream ss;
        for (const auto & kv : _rules) {
            ss << kv.first << " ::= " << kv.second << '\n';
        }
        return ss.str();
    }
};

// Keywords that only annotate a schema, never constrain what it admits.
static const std::unordered_set<std::string> & protected_annotations() {
    static const std::unordered_set<std::string> annotations = {
        "$schema", "title", "description", "$comment", "default", "examples", "deprecated", "readOnly", "writeOnly",
    };
    return annotations;
}

// `propertyNames` constrains object member names.  In JSON every member name is a string, so the two forms
// every schema generator emits - `true` and `{"type": "string"}` - restrict nothing and are accepted here.
// A real key constraint (a pattern, an enum, a length) would have to be spelled into the key rule, which this
// converter does not do, so it stays refused rather than silently widened.
static bool is_vacuous_property_names(const common_json & value) {
    if (value.is_boolean()) {
        return value == true;
    }
    if (!value.is_object()) {
        return false;
    }
    const auto & annotations = protected_annotations();
    for (const auto & [key, child] : value.items()) {
        if (annotations.count(key)) continue;
        if (key != "type") return false;
        if (!child.is_string() || child.get<std::string>() != "string") return false;
    }
    return true;
}

// Which types a type-specific keyword constrains.  JSON Schema says such a keyword only applies to values of
// its own type, so a keyword next to a type it does not belong to constrains nothing.
static bool keyword_applies_to_type(const std::string & key, const std::string & type) {
    static const std::unordered_map<std::string, std::unordered_set<std::string>> by_keyword = {
        {"minLength", {"string"}},
        {"maxLength", {"string"}},
        {"pattern", {"string"}},
        {"format", {"string"}},
        {"minimum", {"integer", "number"}},
        {"maximum", {"integer", "number"}},
        {"exclusiveMinimum", {"integer", "number"}},
        {"exclusiveMaximum", {"integer", "number"}},
        {"multipleOf", {"integer", "number"}},
        {"items", {"array"}},
        {"minItems", {"array"}},
        {"maxItems", {"array"}},
        {"properties", {"object"}},
        {"required", {"object"}},
        {"additionalProperties", {"object"}},
        {"propertyNames", {"object"}},
    };
    const auto it = by_keyword.find(key);
    return it == by_keyword.end() || it->second.count(type) > 0;
}

static void validate_protected_schema(const common_json & schema, const std::string & path = "#") {
    if (!schema.is_object()) {
        throw std::invalid_argument(path + ": schema must be an object");
    }
    const auto & annotations = protected_annotations();

    // A union is admitted when every branch is itself expressible: the converter spells it as one
    // alternative per branch, so the guarantee holds if it holds for each.  A declared `type` next to an
    // `anyOf` is refused rather than dropped - the converter reads the union first, so the type would
    // otherwise vanish and widen what the schema admits.
    if (schema.contains("anyOf")) {
        for (const auto & [key, value] : schema.items()) {
            if (annotations.count(key) || key == "anyOf") continue;
            throw std::invalid_argument(path + ": anyOf cannot be combined with '" + key + "' in strict tool mode");
        }
        const common_json & branches = schema.at("anyOf");
        if (!branches.is_array() || branches.empty()) {
            throw std::invalid_argument(path + "/anyOf must be a non-empty array");
        }
        size_t i = 0;
        for (const auto & branch : branches) {
            validate_protected_schema(branch, path + "/anyOf/" + std::to_string(i++));
        }
        return;
    }

    // {"type": ["a", "b"], ...} is a union of one schema per type, which is how a nullable field is
    // written; validate each alternative exactly as the parser will build it.
    if (schema.contains("type") && schema.at("type").is_array()) {
        const common_json & types = schema.at("type");
        if (types.empty()) {
            throw std::invalid_argument(path + ": type must not be empty");
        }
        size_t i = 0;
        for (const auto & entry : types) {
            if (!entry.is_string()) {
                throw std::invalid_argument(path + ": type must be a string or an array of strings");
            }
            const std::string branch_type = entry.get<std::string>();
            common_json alternative = schema;
            alternative["type"] = entry;
            // In JSON Schema a keyword only constrains values of the type it belongs to, so `minimum` next
            // to `"null"` means nothing - and Zod writes exactly that for `z.number().min(1).nullable()`.
            // Dropping such a keyword from the branch is exact, not a widening; a keyword no type claims is
            // left in place so the checks below still refuse it.
            for (const char * key : {"minLength", "maxLength", "pattern", "format", "minimum", "maximum",
                                     "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "items", "minItems",
                                     "maxItems", "properties", "required", "additionalProperties", "propertyNames"}) {
                if (alternative.contains(key) && !keyword_applies_to_type(key, branch_type)) {
                    alternative.erase(key);
                }
            }
            validate_protected_schema(alternative, path + "/type/" + std::to_string(i++));
        }
        return;
    }

    for (const auto & [key, value] : schema.items()) {
        (void) value;
        if (annotations.count(key)) continue;
        if (key == "$ref" || key == "oneOf" || key == "allOf" || key == "not" || key == "if" ||
            key == "then" || key == "else" || key == "patternProperties" ||
            key == "contains" || key == "prefixItems" || key == "unevaluatedProperties" ||
            key == "unevaluatedItems" || key == "dependentSchemas" || key == "dependencies" || key == "multipleOf" ||
            key == "minProperties" || key == "maxProperties" || key == "minContains" || key == "maxContains") {
            throw std::invalid_argument(path + ": keyword '" + key + "' is not supported in strict tool mode");
        }
        static const std::unordered_set<std::string> supported = {
            "type", "enum", "const", "properties", "required", "additionalProperties", "propertyNames",
            "anyOf", "items", "minItems", "maxItems", "minLength", "maxLength", "pattern", "format",
            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        };
        if (!supported.count(key)) {
            throw std::invalid_argument(path + ": unknown keyword '" + key + "' is not supported in strict tool mode");
        }
    }

    if (schema.contains("$ref")) {
        throw std::invalid_argument(path + ": $ref is not supported in strict tool mode");
    }
    if (schema.contains("enum") || schema.contains("const")) {
        if (schema.contains("enum") && schema.contains("const")) {
            throw std::invalid_argument(path + ": enum and const cannot be combined in strict tool mode");
        }
        for (const auto & [key, value] : schema.items()) {
            if (!annotations.count(key) && key != "type" && key != "enum" && key != "const") {
                throw std::invalid_argument(path + ": enum/const cannot be combined with '" + key + "' in strict tool mode");
            }
        }
        const common_json & values = schema.contains("enum") ? schema.at("enum") : schema.at("const");
        if (schema.contains("enum") && (!values.is_array() || values.empty())) {
            throw std::invalid_argument(path + ": enum must be a non-empty array");
        }
        auto value_type = [](const common_json & value) {
            if (value.is_string()) return std::string("string");
            if (value.is_boolean()) return std::string("boolean");
            if (value.is_null()) return std::string("null");
            if (value.is_array()) return std::string("array");
            if (value.is_object()) return std::string("object");
            if (value.is_number_integer()) return std::string("integer");
            if (value.is_number()) return std::string("number");
            return std::string("invalid");
        };
        std::string expected = schema.contains("enum") ? value_type(values.at(0)) : value_type(values);
        if (expected == "invalid") throw std::invalid_argument(path + ": invalid enum/const value");
        if (schema.contains("enum")) {
            for (const auto & value : values) {
                if (value_type(value) != expected) {
                    throw std::invalid_argument(path + ": mixed-type enums are not supported in strict tool mode");
                }
            }
        }
        if (schema.contains("type")) {
            if (!schema.at("type").is_string()) throw std::invalid_argument(path + ": type must be a string");
            const std::string declared = schema.at("type").get<std::string>();
            if (declared != expected && !(declared == "number" && expected == "integer")) {
                throw std::invalid_argument(path + ": enum/const values do not match the declared type");
            }
        }
        return;
    }

    if (!schema.contains("type") || !schema.at("type").is_string()) {
        throw std::invalid_argument(path + ": strict tool schemas require a concrete string type");
    }
    const std::string type = schema.at("type").get<std::string>();
    auto only = [&](std::initializer_list<const char *> keys) {
        for (const auto & [key, value] : schema.items()) {
            (void) value;
            if (annotations.count(key)) continue;
            if (std::find_if(keys.begin(), keys.end(), [&](const char * allowed) { return key == allowed; }) == keys.end()) {
                throw std::invalid_argument(path + ": keyword '" + key + "' is invalid for type " + type);
            }
        }
    };
    if (type == "string") {
        only({"type", "minLength", "maxLength", "pattern", "format"});
        if (schema.contains("pattern") && !schema.at("pattern").is_string()) {
            throw std::invalid_argument(path + ": pattern must be a string");
        }
        if (schema.contains("pattern")) {
            // Only the converter's own limit is prechecked; whether the pattern NEEDS a '<' (which a
            // protected value cannot contain) is decided by the class analysis in _visit_pattern, because
            // a pattern can spell '<' directly, as an escape, or exclude it and be perfectly safe.
            const std::string pattern = schema.at("pattern").get<std::string>();
            if (pattern.size() < 2 || pattern.front() != '^' || pattern.back() != '$') {
                throw std::invalid_argument(path + ": pattern must be anchored with '^' and '$' in strict tool mode");
            }
        }
        if (schema.contains("format")) {
            const common_json & format = schema.at("format");
            if (!format.is_string()) {
                throw std::invalid_argument(path + ": format must be a string");
            }
            static const std::unordered_set<std::string> formats = {"date", "time", "date-time", "uuid"};
            if (!formats.count(format.get<std::string>())) {
                throw std::invalid_argument(path + ": format '" + format.get<std::string>() +
                                            "' is not supported in strict tool mode (use date, time, date-time or uuid)");
            }
        }
        for (const char * key : {"minLength", "maxLength"}) {
            if (schema.contains(key) && (!schema.at(key).is_number_integer() || schema.at(key).get<int64_t>() < 0)) {
                throw std::invalid_argument(path + ": string length bounds must be non-negative integers");
            }
        }
        if (schema.contains("minLength") && schema.contains("maxLength") && schema.at("minLength").get<int64_t>() > schema.at("maxLength").get<int64_t>()) {
            throw std::invalid_argument(path + ": minLength exceeds maxLength");
        }
        return;
    }
    if (type == "object") {
        only({"type", "properties", "required", "additionalProperties", "propertyNames"});
        if (schema.contains("propertyNames") && !is_vacuous_property_names(schema.at("propertyNames"))) {
            throw std::invalid_argument(path + ": propertyNames is only supported as {\"type\": \"string\"} in strict tool mode");
        }
        if (schema.contains("properties") && !schema.at("properties").is_object()) {
            throw std::invalid_argument(path + "/properties must be an object");
        }
        // `properties: {}` is what schema generators emit for a bare dictionary; it declares no names, so
        // the key rule stays the string primitive and there is nothing to reject.
        const bool has_properties = schema.contains("properties") && !schema.at("properties").empty();
        if (schema.contains("additionalProperties") && schema.at("additionalProperties") != false) {
            const common_json & extra = schema.at("additionalProperties");
            // an empty schema ({"type": "object"} from `record<string, unknown>`) means "any value", which
            // the protected `value` primitive spells without ever emitting a raw '<'
            if (!extra.is_object()) {
                throw std::invalid_argument(path + ": additionalProperties must be false, an empty schema or a typed schema in strict tool mode");
            }
            if (has_properties) {
                // a dictionary with declared properties too would need a key rule that rejects the declared
                // names, and that rule is built from a bare character class which admits a raw '<'
                throw std::invalid_argument(path + ": a dictionary cannot be combined with declared properties in strict tool mode");
            }
            if (!extra.empty()) {
                validate_protected_schema(extra, path + "/additionalProperties");
            }
        }
        if (has_properties) {
            for (const auto & [name, child] : schema.at("properties").items()) {
                validate_protected_schema(child, path + "/properties/" + name);
            }
        }
        if (schema.contains("required")) {
            const auto & required = schema.at("required");
            if (!required.is_array()) throw std::invalid_argument(path + "/required must be an array");
            for (const auto & key : required) {
                if (!key.is_string() || !schema.contains("properties") || !schema.at("properties").contains(key.get<std::string>())) {
                    throw std::invalid_argument(path + ": required names must refer to declared properties");
                }
            }
        }
        return;
    }
    if (type == "array") {
        only({"type", "items", "minItems", "maxItems"});
        if (!schema.contains("items")) throw std::invalid_argument(path + ": array schemas require an items schema in strict tool mode");
        // `items: {}` is a free-form element, which the protected `value` primitive spells without ever
        // emitting a raw '<'
        if (!schema.at("items").is_object()) throw std::invalid_argument(path + "/items must be an object");
        if (!schema.at("items").empty()) validate_protected_schema(schema.at("items"), path + "/items");
        for (const char * key : {"minItems", "maxItems"}) {
            if (schema.contains(key) && (!schema.at(key).is_number_integer() || schema.at(key).get<int64_t>() < 0)) {
                throw std::invalid_argument(path + ": item bounds must be non-negative integers");
            }
        }
        if (schema.contains("minItems") && schema.contains("maxItems") && schema.at("minItems").get<int64_t>() > schema.at("maxItems").get<int64_t>()) {
            throw std::invalid_argument(path + ": minItems exceeds maxItems");
        }
        return;
    }
    if (type == "integer") {
        only({"type", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"});
        if (schema.contains("minimum") && schema.contains("exclusiveMinimum")) throw std::invalid_argument(path + ": minimum and exclusiveMinimum cannot be combined");
        if (schema.contains("maximum") && schema.contains("exclusiveMaximum")) throw std::invalid_argument(path + ": maximum and exclusiveMaximum cannot be combined");
        // the schema builder rounds every bound towards the integers it still admits, so a value it
        // cannot represent (not a number, not finite, past int64) must be refused here, before rounding
        auto bound = [&](const char * key) {
            const common_json & value = schema.at(key);
            if (!value.is_number()) throw std::invalid_argument(path + ": " + key + " must be a number");
            const double d = value.is_number_integer() ? (double) value.get<int64_t>() : value.get<double>();
            if (!std::isfinite(d) || d <= -9.2e18 || d >= 9.2e18) {
                throw std::invalid_argument(path + ": " + key + " is outside the integer range this converter can spell");
            }
            return d;
        };
        const bool has_min = schema.contains("minimum") || schema.contains("exclusiveMinimum");
        const bool has_max = schema.contains("maximum") || schema.contains("exclusiveMaximum");
        const double lo = schema.contains("minimum") ? bound("minimum") : schema.contains("exclusiveMinimum") ? bound("exclusiveMinimum") + 1.0 : 0.0;
        const double hi = schema.contains("maximum") ? bound("maximum") : schema.contains("exclusiveMaximum") ? bound("exclusiveMaximum") - 1.0 : 0.0;
        if (has_min && has_max && lo > hi) {
            throw std::invalid_argument(path + ": the integer bounds admit no value");
        }
        return;
    }
    if (type == "number" || type == "boolean" || type == "null") {
        only({"type"});
        return;
    }
    throw std::invalid_argument(path + ": unsupported type '" + type + "' in strict tool mode");
}

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf, const common_grammar_options & options) {
#ifdef LLAMA_USE_LLGUIDANCE
    if (!force_gbnf && !options.tool_value && options.forbidden_sequence.empty()) {
        return "%llguidance {}\nstart: %json " + schema.dump();
    }
#else
    (void) force_gbnf;
#endif // LLAMA_USE_LLGUIDANCE
    try {
        if (options.tool_value) {
            validate_protected_schema(schema);
        }
        common_chat_schema_converter converter(false, options.tool_value, options.forbidden_sequence);
        converter.visit(*common_chat_schema_from_json(schema).root, "");
        converter.check_errors();
        return converter.format_grammar();
    } catch (const std::runtime_error & e) {
        throw std::invalid_argument(std::string("JSON schema conversion failed:\n") + e.what());
    }
}

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf) {
    return json_schema_to_grammar(schema, force_gbnf, {});
}

std::string json_schema_to_grammar(const common_chat_schema_document & schema) {
    common_chat_schema_converter converter(false);
    converter.visit(*schema.root, "");
    converter.check_errors();
    return converter.format_grammar();
}

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options) {
    common_chat_schema_converter converter(options.dotall, options.tool_value, options.forbidden_sequence);
    common_grammar_builder builder {
        /* .add_rule = */ [&](const std::string & name, const std::string & rule) {
            return converter._add_rule(name, rule);
        },
        /* .add_schema = */ [&](const std::string & name, const common_chat_schema & schema) {
            return converter.add_schema(name == "root" ? "" : name, schema);
        },
    };
    cb(builder);
    converter.check_errors();
    return converter.format_grammar();
}
