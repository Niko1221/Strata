// Vendored verbatim from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (common/json-schema.cpp), MIT license (see LICENSE-llama.txt in this directory).
// Strata edits are marked with "STRATA:".
// They add the strict-mode keyword audit and the bound validation; the upstream conversion is unchanged.
#include "json-schema.h"
#include "json_compat.h"  // STRATA: shim for llama.cpp common.h - string_split only

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class common_chat_schema_builder {
    const common_json &      root_;       // the schema being built
    const common_json &      ref_root_;    // STRATA: the document a $ref resolves against (== root_ normally)
    common_chat_schema_document & doc_;
    // STRATA: report a keyword the built node does not express (strict tool mode); nullptr refuses instead
    std::vector<std::string> * dropped_ = nullptr;
    bool                     strict_ = false;

    // the targets built here, moved into doc_ once the whole schema is built
    std::map<std::string, common_chat_schema_ptr> refs_;

    // ref nodes get their target once every $ref is built, a cycle would otherwise need it too early
    std::vector<common_chat_schema_ref *> pending_;

    [[noreturn]] static void fail(const std::string & path, const std::string & msg) {
        throw std::runtime_error("JSON schema error at " + path + ": " + msg);
    }

    // STRATA: the keywords each node kind actually expresses.  A branch below reads one keyword at a time,
    // and anything else in that subschema is dropped without a word - {"type":"number","minimum":1} converts
    // to an unbounded number, a tuple loses minItems/maxItems.  Keeping the list inside the builder puts it
    // next to the reads it describes, so a branch that starts reading a keyword has its list in the same
    // function.  A keyword left over is REPORTED (strict tool mode turns it into a warning) or, without a
    // sink, refused: every one of these constrains the value only, never the surrounding envelope, so losing
    // it widens the language without breaking the delimiter guarantee.  `type` is always accepted: it
    // selected the branch.
    void audit_keywords(const common_json & schema, const std::string & path, common_chat_schema::node_kind kind) {
        // Only what really is an annotation: contentEncoding/contentMediaType constrain the value (base64,
        // JSON-in-a-string), so they are reported like any other unexpressed constraint instead of being
        // waved through as documentation.
        static const std::unordered_set<std::string> annotations = {
            "$schema", "$id", "$comment", "$anchor", "$defs", "definitions", "title", "description",
            "default", "examples", "deprecated", "readOnly", "writeOnly",
        };
        std::unordered_set<std::string> consumed;
        switch (kind) {
            case common_chat_schema::KIND_REF:     consumed = {"$ref"}; break;
            case common_chat_schema::KIND_ANY_OF:  consumed = {"oneOf", "anyOf"}; break;
            case common_chat_schema::KIND_ALL_OF:  consumed = {"allOf"}; break;
            case common_chat_schema::KIND_CONST:   consumed = {"const"}; break;
            case common_chat_schema::KIND_ENUM:    consumed = {"enum"}; break;
            case common_chat_schema::KIND_INTEGER: consumed = {"minimum", "maximum", "exclusiveMinimum",
                                                              "exclusiveMaximum"}; break;
            case common_chat_schema::KIND_STRING:  consumed = {"pattern", "format", "minLength", "maxLength"}; break;
            case common_chat_schema::KIND_ARRAY:   consumed = {"items", "minItems", "maxItems"}; break;
            // a tuple's length is its item list, so a count beside it has nowhere to go
            case common_chat_schema::KIND_TUPLE:   consumed = {"items", "prefixItems"}; break;
            // propertyNames is read by the strict-mode preflight, which admits only the vacuous spellings
            case common_chat_schema::KIND_OBJECT:  consumed = {"properties", "required", "additionalProperties",
                                                              "propertyNames"}; break;
            default: break;   // a leaf with no keywords of its own, or a union whose branches are audited in turn
        }
        for (const auto & [key, value] : schema.items()) {
            (void) value;
            if (annotations.count(key) || key == "type" || consumed.count(key)) continue;
            // A keyword that does not constrain this node's type at all is a no-op in JSON Schema - not a
            // loss, and saying so would bury the real notes in noise.  Everything else constrains the value,
            // and this node cannot spell it.
            const std::string node_type = type_name_of(kind);
            if (!node_type.empty() && !keyword_applies_to_type(key, node_type)) continue;
            const std::string note = "keyword '" + key + "' at " + path + " is not expressed by " +
                                     common_chat_schema::kind_name(kind) + " and was dropped";
            if (dropped_ != nullptr) {
                dropped_->push_back(note);
                continue;
            }
            fail(path, note);
        }
    }

    // the `type` spelling of a JSON value, for the exact `enum`-beside-`type` narrowing below
    static std::string json_type_name(const common_json & value) {
        if (value.is_string())       return "string";
        if (value.is_boolean())      return "boolean";
        if (value.is_null())         return "null";
        if (value.is_array())        return "array";
        if (value.is_object())       return "object";
        if (value.is_number_integer()) return "integer";
        if (value.is_number())       return "number";
        return "invalid";
    }

    // the `type` spelling of a built node, or "" for a kind no keyword map speaks about
    static std::string type_name_of(common_chat_schema::node_kind kind) {
        switch (kind) {
            case common_chat_schema::KIND_NULL:    return "null";
            case common_chat_schema::KIND_BOOLEAN: return "boolean";
            case common_chat_schema::KIND_NUMBER:  return "number";
            case common_chat_schema::KIND_INTEGER: return "integer";
            case common_chat_schema::KIND_STRING:  return "string";
            case common_chat_schema::KIND_ARRAY:
            case common_chat_schema::KIND_TUPLE:   return "array";
            case common_chat_schema::KIND_OBJECT:  return "object";
            default: return "";
        }
    }

    static int get_count(const common_json & schema, const std::string & key, const std::string & path, int def) {
        if (!schema.contains(key)) {
            return def;
        }
        const common_json & value = schema.at(key);
        if (!value.is_number_integer() || value.get<int>() < 0) {
            fail(path, key + " must be a non-negative integer");
        }
        return value.get<int>();
    }

    // a fractional bound is rounded inwards, towards the integers it still admits; `adjust` is the +1/-1 an
    // exclusive bound needs, applied here so the range check covers it
    static int64_t get_bound(const common_json & schema, const std::string & key, const std::string & path,
                             bool round_up, int64_t adjust = 0) {
        const common_json & value = schema.at(key);
        // 2^63 is exact as a double and is one past the largest int64, so the upper test is `>=`
        constexpr double two_pow_63 = 9223372036854775808.0;
        auto check = [&](double bound) {
            // Out of range (or adjusted past it) would make the cast, or the +/-1 an exclusive bound adds,
            // undefined - the grammar would spell a different range than the schema asks for.
            if (!std::isfinite(bound) || bound < -two_pow_63 || bound >= two_pow_63) {
                fail(path, key + " is outside the supported integer range");
            }
            return (int64_t) bound;
        };
        if (value.is_number_integer()) {
            const int64_t bound = value.get<int64_t>();
            if ((adjust > 0 && bound > std::numeric_limits<int64_t>::max() - adjust) ||
                (adjust < 0 && bound < std::numeric_limits<int64_t>::min() - adjust)) {
                fail(path, key + " is outside the supported integer range");
            }
            return bound + adjust;
        }
        if (!value.is_number()) {
            fail(path, key + " must be a number");
        }
        const double d = value.get<double>();
        const double rounded = round_up ? std::ceil(d) : std::floor(d);
        return check(rounded + (double) adjust);
    }

    static common_chat_schema::string_format get_format(const common_json & schema, const std::string & path) {
        if (!schema.contains("format")) {
            return common_chat_schema::FORMAT_NONE;
        }
        const common_json & value = schema.at("format");
        if (!value.is_string()) {
            fail(path, "format must be a string");
        }
        std::string format = value.get<std::string>();
        if (format == "date") {
            return common_chat_schema::FORMAT_DATE;
        }
        if (format == "time") {
            return common_chat_schema::FORMAT_TIME;
        }
        if (format == "date-time") {
            return common_chat_schema::FORMAT_DATE_TIME;
        }
        if (format == "uuid" || (format.size() == 5 && format.compare(0, 4, "uuid") == 0 && format[4] >= '1' && format[4] <= '5')) {
            return common_chat_schema::FORMAT_UUID;
        }
        return common_chat_schema::FORMAT_NONE;
    }

    const common_json & resolve_ref(const std::string & ref, const std::string & path) {
        // A JSON Pointer is rooted at the DOCUMENT, which is not necessarily the node being built: the tool
        // builder compiles one parameter at a time, and its `#/properties/x` still names a sibling.
        const common_json * target = &ref_root_;
        auto tokens = string_split(ref.substr(1), "/");
        for (size_t i = 1; i < tokens.size(); i++) {
            const std::string & sel = tokens[i];
            if (target->is_object() && target->contains(sel)) {
                target = &target->at(sel);
            } else if (target->is_array()) {
                size_t idx;
                try {
                    idx = std::stoull(sel);
                } catch (const std::logic_error &) {
                    idx = target->size();
                }
                if (idx >= target->size()) {
                    fail(path, "cannot resolve $ref " + ref + ", " + sel + " is out of range");
                }
                target = &target->at(idx);
            } else {
                fail(path, "cannot resolve $ref " + ref + ", " + sel + " not found");
            }
        }
        return *target;
    }

    common_chat_schema_ptr build_ref(const common_json & value, const std::string & path) {
        if (!value.is_string()) {
            fail(path, "$ref must be a string");
        }
        std::string ref = value.get<std::string>();
        if (ref.compare(0, 2, "#/") != 0) {
            fail(path, "unsupported $ref " + ref + ", only references into the same document are supported");
        }
        if (refs_.find(ref) == refs_.end()) {
            // reserve the key first, so that a cycle back to this $ref stops here
            refs_[ref] = nullptr;
            refs_[ref] = build_node(resolve_ref(ref, path), ref);
        }
        auto node = std::make_unique<common_chat_schema_ref>(ref);
        pending_.push_back(node.get());
        return node;
    }

    template <typename T>
    common_chat_schema_ptr build_alternatives(const common_json & alts, const std::string & path) {
        if (!alts.is_array()) {
            fail(path, "must be an array of schemas");
        }
        if (alts.empty()) {
            fail(path, "must not be empty");
        }
        auto node = std::make_unique<T>();
        size_t i = 0;
        for (const auto & alt : alts) {
            node->children.push_back(build_node(alt, path + "/" + std::to_string(i++)));
        }
        return node;
    }

    common_chat_schema_ptr build_object(const common_json & schema, const std::string & path) {
        auto node = std::make_unique<common_chat_schema_object>();

        std::unordered_set<std::string> required;
        if (schema.contains("required") && schema.at("required").is_array()) {
            for (const auto & name : schema.at("required")) {
                if (name.is_string()) {
                    required.insert(name.get<std::string>());
                }
            }
        }

        if (schema.contains("properties")) {
            const common_json & properties = schema.at("properties");
            if (!properties.is_object()) {
                fail(path, "properties must be an object");
            }
            for (const auto & [name, prop] : properties.items()) {
                node->properties.push_back({name, build_node(prop, path + "/properties/" + name), required.count(name) > 0});
            }
        }

        if (schema.contains("additionalProperties")) {
            const common_json & additional = schema.at("additionalProperties");
            if (additional.is_boolean()) {
                if (additional.get<bool>()) {
                    node->additional_properties = std::make_unique<common_chat_schema_any>();
                }
            } else if (additional.is_object()) {
                node->additional_properties = build_node(additional, path + "/additionalProperties");
            } else {
                fail(path, "additionalProperties must be a boolean or a schema");
            }
        } else if (!schema.contains("properties")) {
            // {"type": "object"} on its own accepts any object
            node->additional_properties = std::make_unique<common_chat_schema_any>();
        }

        return node;
    }

    common_chat_schema_ptr build_array(const common_json & schema, const std::string & path) {
        auto node = std::make_unique<common_chat_schema_array>();
        if (schema.contains("items") || schema.contains("prefixItems")) {
            // "items" wins when both are present; as in the converter, a schema instead of an array is the item schema
            const std::string key = schema.contains("items") ? "items" : "prefixItems";
            const common_json & items = schema.at(key);
            if (items.is_array()) {
                auto tuple = std::make_unique<common_chat_schema_tuple>();
                size_t i = 0;
                for (const auto & item : items) {
                    tuple->items.push_back(build_node(item, path + "/" + key + "/" + std::to_string(i++)));
                }
                return tuple;
            }
            node->items = build_node(items, path + "/" + key);
        } else {
            node->items = std::make_unique<common_chat_schema_any>();
        }
        node->min_items = get_count(schema, "minItems", path, 0);
        node->max_items = get_count(schema, "maxItems", path, -1);
        return node;
    }

    common_chat_schema_ptr build_string(const common_json & schema, const std::string & path) {
        auto node = std::make_unique<common_chat_schema_string>();
        if (schema.contains("pattern")) {
            const common_json & pattern = schema.at("pattern");
            if (!pattern.is_string()) {
                fail(path, "pattern must be a string");
            }
            node->pattern = pattern.get<std::string>();
        }
        node->format     = get_format(schema, path);
        node->min_length = get_count(schema, "minLength", path, 0);
        node->max_length = get_count(schema, "maxLength", path, -1);
        return node;
    }

    common_chat_schema_ptr build_integer(const common_json & schema, const std::string & path) {
        auto node = std::make_unique<common_chat_schema_integer>();
        if (schema.contains("minimum")) {
            node->minimum = get_bound(schema, "minimum", path, /* round_up */ true);
        } else if (schema.contains("exclusiveMinimum")) {
            node->minimum = get_bound(schema, "exclusiveMinimum", path, /* round_up */ false, /* adjust */ 1);
        }
        if (schema.contains("maximum")) {
            node->maximum = get_bound(schema, "maximum", path, /* round_up */ false);
        } else if (schema.contains("exclusiveMaximum")) {
            node->maximum = get_bound(schema, "exclusiveMaximum", path, /* round_up */ true, /* adjust */ -1);
        }
        return node;
    }

    common_chat_schema_ptr build_node(const common_json & schema, const std::string & path) {
        common_chat_schema_ptr node = build_node_value(schema, path);
        // A schema whose `type` is an array becomes one branch per type, and each branch is audited on its
        // own; the union node itself expresses none of the schema's keywords, so auditing it would report
        // exactly the keywords the branches do honour.
        const bool union_of_types = schema.is_object() && schema.contains("type") && schema.at("type").is_array();
        if (strict_ && !union_of_types) {
            audit_keywords(schema, path, node->kind());
        }
        return node;
    }

    common_chat_schema_ptr build_node_value(const common_json & schema, const std::string & path) {
        if (!schema.is_object()) {
            fail(path, "schema must be an object");
        }
        if (schema.contains("$ref")) {
            return build_ref(schema.at("$ref"), path);
        }
        if (schema.contains("oneOf") || schema.contains("anyOf")) {
            const std::string key = schema.contains("oneOf") ? "oneOf" : "anyOf";
            return build_alternatives<common_chat_schema_any_of>(schema.at(key), path + "/" + key);
        }

        common_json type;
        if (schema.contains("type")) {
            type = schema.at("type");
        }
        if (type.is_array()) {
            // {"type": ["a", "b"], ...} is {"anyOf": [{"type": "a", ...}, {"type": "b", ...}]}
            if (type.empty()) {
                fail(path, "type must not be empty");
            }
            auto node = std::make_unique<common_chat_schema_any_of>();
            size_t i = 0;
            for (const auto & t : type) {
                common_json alt = schema;
                alt["type"] = t;
                if (strict_ && t.is_string()) {
                    // A keyword only constrains values of the type it belongs to, so `minimum` beside "null"
                    // means nothing - Zod writes exactly that for z.number().min(1).nullable(). Dropping it
                    // from that branch is exact, not a widening; the branch's own audit decides the rest.
                    for (const char * key : {"minLength", "maxLength", "pattern", "format", "minimum", "maximum",
                                             "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "items",
                                             "minItems", "maxItems", "properties", "required",
                                             "additionalProperties", "propertyNames"}) {
                        if (alt.contains(key) && !keyword_applies_to_type(key, t.get<std::string>())) {
                            alt.erase(key);
                        }
                    }
                }
                node->children.push_back(build_node(alt, path + "/type/" + std::to_string(i++)));
            }
            return node;
        }
        if (schema.contains("const")) {
            return std::make_unique<common_chat_schema_const>(schema.at("const"));
        }
        if (schema.contains("enum")) {
            const common_json & values = schema.at("enum");
            if (!values.is_array() || values.empty()) {
                fail(path, "enum must be a non-empty array");
            }
            auto node = std::make_unique<common_chat_schema_enum>();
            // STRATA (strict tool mode only): `enum` beside `type` is an intersection, so a value outside the
            // declared type is unsatisfiable and dropping it is EXACT - not a widening.  That keeps a mixed
            // enum such as {"type":"string","enum":["a",1]} convertible: the grammar spells `a` and the reply
            // parser then decodes it with the declared type, so a surviving value always has the right type.
            const std::string narrow_to = strict_ && type.is_string() ? type.get<std::string>() : std::string();
            for (const auto & value : values) {
                if (!narrow_to.empty() && json_type_name(value) != narrow_to &&
                    !(narrow_to == "number" && json_type_name(value) == "integer")) {
                    continue;
                }
                node->values.push_back(value);
            }
            if (node->values.empty()) {
                fail(path, "no enum value matches the declared type " + narrow_to);
            }
            return node;
        }
        if (!type.is_null() && !type.is_string()) {
            fail(path, "type must be a string or an array of strings");
        }

        const std::string type_name = type.is_string() ? type.get<std::string>() : "";
        const bool has_properties = schema.contains("properties") ||
            (schema.contains("additionalProperties") && schema.at("additionalProperties") != true);

        if (type_name.empty()) {
            // without a type the structural keywords decide, in the same order as the converter
            if (has_properties) {
                return build_object(schema, path);
            }
            if (schema.contains("allOf")) {
                return build_alternatives<common_chat_schema_all_of>(schema.at("allOf"), path + "/allOf");
            }
            if (schema.contains("items") || schema.contains("prefixItems")) {
                return build_array(schema, path);
            }
            if (schema.contains("pattern") || schema.contains("minLength") || schema.contains("maxLength") || get_format(schema, path) != common_chat_schema::FORMAT_NONE) {
                return build_string(schema, path);
            }
            return std::make_unique<common_chat_schema_any>();
        }
        if (type_name == "object") {
            if (!has_properties && schema.contains("allOf")) {
                return build_alternatives<common_chat_schema_all_of>(schema.at("allOf"), path + "/allOf");
            }
            return build_object(schema, path);
        }
        if (type_name == "string") {
            if (schema.contains("allOf")) {
                return build_alternatives<common_chat_schema_all_of>(schema.at("allOf"), path + "/allOf");
            }
            return build_string(schema, path);
        }
        if (type_name == "array") {
            return build_array(schema, path);
        }
        if (type_name == "integer") {
            return build_integer(schema, path);
        }
        if (type_name == "number") {
            return std::make_unique<common_chat_schema_number>();
        }
        if (type_name == "boolean") {
            return std::make_unique<common_chat_schema_boolean>();
        }
        if (type_name == "null") {
            return std::make_unique<common_chat_schema_null>();
        }
        fail(path, "unrecognized type " + type_name);
    }

  public:
    common_chat_schema_builder(const common_json & root, common_chat_schema_document & doc, bool strict = false,
                               std::vector<std::string> * dropped = nullptr,
                               const common_json * ref_root = nullptr)
        : root_(root), ref_root_(ref_root != nullptr ? *ref_root : root), doc_(doc),
          dropped_(dropped), strict_(strict) {}

    common_chat_schema_ptr build() {
        auto node = build_node(root_, "#");
        for (auto & entry : refs_) {
            doc_.refs[entry.first] = std::move(entry.second);
        }
        for (auto * ref : pending_) {
            ref->target = doc_.refs.at(ref->ref).get();
        }
        return node;
    }
};

bool keyword_applies_to_type(const std::string & key, const std::string & type) {
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
        {"prefixItems", {"array"}},
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

common_chat_schema_document common_chat_schema_from_json(const common_json & schema, bool strict,
                                                         std::vector<std::string> * dropped,
                                                         const common_json * ref_root) {
    common_chat_schema_document doc;
    doc.root = common_chat_schema_builder(schema, doc, strict, dropped, ref_root).build();
    return doc;
}

static common_chat_schema::value_type json_type(const common_json & value) {
    if (value.is_null()) {
        return common_chat_schema::TYPE_NULL;
    }
    if (value.is_boolean()) {
        return common_chat_schema::TYPE_BOOLEAN;
    }
    if (value.is_number_integer()) {
        return common_chat_schema::TYPE_INTEGER;
    }
    if (value.is_number()) {
        return common_chat_schema::TYPE_NUMBER;
    }
    if (value.is_string()) {
        return common_chat_schema::TYPE_STRING;
    }
    if (value.is_array()) {
        return common_chat_schema::TYPE_ARRAY;
    }
    return common_chat_schema::TYPE_OBJECT;
}

static common_chat_schema::type_set value_types_impl(const common_chat_schema & s, std::unordered_set<const common_chat_schema *> & visited) {
    switch (s.kind()) {
        case common_chat_schema::KIND_ANY:
            return common_chat_schema::type_set::all();
        case common_chat_schema::KIND_NULL:
            return { common_chat_schema::TYPE_NULL };
        case common_chat_schema::KIND_BOOLEAN:
            return { common_chat_schema::TYPE_BOOLEAN };
        case common_chat_schema::KIND_NUMBER:
            return { common_chat_schema::TYPE_NUMBER, common_chat_schema::TYPE_INTEGER };
        case common_chat_schema::KIND_INTEGER:
            return { common_chat_schema::TYPE_INTEGER };
        case common_chat_schema::KIND_STRING:
            return { common_chat_schema::TYPE_STRING };
        case common_chat_schema::KIND_ARRAY:
        case common_chat_schema::KIND_TUPLE:
            return { common_chat_schema::TYPE_ARRAY };
        case common_chat_schema::KIND_OBJECT:
            return { common_chat_schema::TYPE_OBJECT };
        case common_chat_schema::KIND_CONST:
            return { json_type(static_cast<const common_chat_schema_const &>(s).value) };
        case common_chat_schema::KIND_ENUM: {
            common_chat_schema::type_set types;
            for (const auto & value : static_cast<const common_chat_schema_enum &>(s).values) {
                types.add(json_type(value));
            }
            return types;
        }
        case common_chat_schema::KIND_REF: {
            const auto * target = static_cast<const common_chat_schema_ref &>(s).target;
            if (!target || !visited.insert(target).second) {
                // a cycle contributes no type, to be safe
                return {};
            }
            auto types = value_types_impl(*target, visited);
            visited.erase(target);
            return types;
        }
        case common_chat_schema::KIND_ANY_OF: {
            common_chat_schema::type_set types;
            for (const auto & child : static_cast<const common_chat_schema_any_of &>(s).children) {
                types |= value_types_impl(*child, visited);
            }
            return types;
        }
        case common_chat_schema::KIND_ALL_OF: {
            auto types = common_chat_schema::type_set::all();
            for (const auto & child : static_cast<const common_chat_schema_all_of &>(s).children) {
                types &= value_types_impl(*child, visited);
            }
            return types;
        }
    }
    return {};
}

common_chat_schema::type_set common_chat_schema::value_types() const {
    std::unordered_set<const common_chat_schema *> visited;
    return value_types_impl(*this, visited);
}

static bool may_be_string_impl(const common_chat_schema & s, std::unordered_set<const common_chat_schema *> & visited) {
    switch (s.kind()) {
        case common_chat_schema::KIND_STRING:
            return true;
        case common_chat_schema::KIND_CONST:
            return static_cast<const common_chat_schema_const &>(s).value.is_string();
        case common_chat_schema::KIND_ENUM:
            for (const auto & v : static_cast<const common_chat_schema_enum &>(s).values) {
                if (v.is_string()) {
                    return true;
                }
            }
            return false;
        case common_chat_schema::KIND_REF: {
            // a cycle is taken as not a string, to be safe
            const auto * target = static_cast<const common_chat_schema_ref &>(s).target;
            if (!target || !visited.insert(target).second) {
                return false;
            }
            bool result = may_be_string_impl(*target, visited);
            visited.erase(target);
            return result;
        }
        case common_chat_schema::KIND_ANY_OF:
            for (const auto & child : static_cast<const common_chat_schema_any_of &>(s).children) {
                if (may_be_string_impl(*child, visited)) {
                    return true;
                }
            }
            return false;
        case common_chat_schema::KIND_ALL_OF: {
            // every child must allow a string, an any child constrains nothing
            bool any_string = false;
            for (const auto & child : static_cast<const common_chat_schema_all_of &>(s).children) {
                if (child->kind() == common_chat_schema::KIND_ANY) {
                    continue;
                }
                if (!may_be_string_impl(*child, visited)) {
                    return false;
                }
                any_string = true;
            }
            return any_string;
        }
        default:
            return false;
    }
}

bool common_chat_schema::may_be_string() const {
    std::unordered_set<const common_chat_schema *> visited;
    return may_be_string_impl(*this, visited);
}

const char * common_chat_schema::kind_name(node_kind kind) {
    switch (kind) {
        case KIND_ANY:     return "any";
        case KIND_REF:     return "ref";
        case KIND_ANY_OF:  return "anyOf";
        case KIND_ALL_OF:  return "allOf";
        case KIND_CONST:   return "const";
        case KIND_ENUM:    return "enum";
        case KIND_NULL:    return "null";
        case KIND_BOOLEAN: return "boolean";
        case KIND_NUMBER:  return "number";
        case KIND_INTEGER: return "integer";
        case KIND_STRING:  return "string";
        case KIND_ARRAY:   return "array";
        case KIND_TUPLE:   return "tuple";
        case KIND_OBJECT:  return "object";
    }
    return "?";
}

const char * common_chat_schema::type_name(value_type type) {
    switch (type) {
        case TYPE_NULL:    return "null";
        case TYPE_BOOLEAN: return "boolean";
        case TYPE_NUMBER:  return "number";
        case TYPE_INTEGER: return "integer";
        case TYPE_STRING:  return "string";
        case TYPE_ARRAY:   return "array";
        case TYPE_OBJECT:  return "object";
    }
    return "?";
}
