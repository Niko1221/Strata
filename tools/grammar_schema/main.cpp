// grammar_schema: read a JSON Schema on stdin, write the equivalent GBNF grammar (root rule "root") on stdout.
// Built from the vendored llama.cpp converter (vendor/json_schema_to_grammar/, MIT).  Exits 1 with a single
// stderr line when the input is not valid JSON or the schema cannot be converted.  force_gbnf=true so the
// converter never emits an %llguidance grammar (Strata's engine has no llguidance backend).
#include "json-schema-to-grammar.h"
#include "json.h"

#include <iostream>
#include <string>

int main() {
    std::string text;
    {
        char buf[65536];
        while (std::cin.read(buf, sizeof buf)) text.append(buf, std::cin.gcount());
        text.append(buf, std::cin.gcount());
    }

    common_json schema;
    try {
        schema = common_json::parse(text);
    } catch (const std::exception & e) {
        std::cerr << "grammar_schema: invalid JSON: " << e.what() << "\n";
        return 1;
    }

    try {
        std::string gbnf = json_schema_to_grammar(schema, /* force_gbnf= */ true);
        std::cout << gbnf;
        if (!gbnf.empty() && gbnf.back() != '\n') std::cout << "\n";
    } catch (const std::exception & e) {
        std::cerr << "grammar_schema: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
