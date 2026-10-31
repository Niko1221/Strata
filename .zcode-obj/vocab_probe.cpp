// vocab_probe: dump the GrammarVocab the engine builds from a GGUF (pieces/control/eog), for
// mask debugging. vocab_probe.exe <shard.gguf> [id1 id2 ...] | vocab_probe.exe <shard.gguf> --dump out.json
#include <cstdio>
#include <string>
#include "strata/grammar/grammar.hpp"
#include "strata/grammar/grammar_vocab.hpp"

using namespace strata::grammar;

static void emit_piece(FILE* f, const std::string& s) {
    fputc('"', f);
    for (unsigned char c : s) {
        if (c == '"') fputs("\\\"", f);
        else if (c == '\\') fputs("\\\\", f);
        else if (c == '\n') fputs("\\n", f);
        else if (c == '\r') fputs("\\r", f);
        else if (c == '\t') fputs("\\t", f);
        else if (c < 32) fprintf(f, "\\u%04x", c);
        else if (c >= 0x80) fprintf(f, "\\u%04x", c);  // decoded byte tokens: raw 0x80+ is not valid UTF-8
        else fputc(c, f);
    }
    fputc('"', f);
}

int main(int argc, char** argv) {
    GrammarVocab v;
    std::string err;
    if (!load_grammar_vocab(argv[1], v, err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    printf("n=%zu control=%zu eog=%zu\n", v.pieces.size(), v.control.size(), v.eog.size());
    for (size_t i = 0; i < v.eog.size(); i++)
        if (v.eog[i]) printf("eog id=%zu piece=%s ctrl=%d\n", i, v.pieces[i].c_str(), (int) v.control[i]);
    if (argc > 3 && std::string(argv[2]) == "--dump") {  // full vocab json for the offline probes
        FILE* f = fopen(argv[3], "wb");
        fputc('[', f);
        for (size_t i = 0; i < v.pieces.size(); i++) {
            if (i) fputc(',', f);
            fputs("{\"p\":", f);
            emit_piece(f, v.pieces[i]);
            fprintf(f, ",\"c\":%d,\"e\":%d}", (int) v.control[i], (int) v.eog[i]);
        }
        fputc(']', f);
        fclose(f);
        printf("dumped %zu pieces to %s\n", v.pieces.size(), argv[3]);
        return 0;
    }
    for (int a = 2; a < argc; a++) {
        int id = atoi(argv[a]);
        if (id >= 0 && (size_t) id < v.pieces.size())
            printf("id=%d piece=%s ctrl=%d eog=%d\n", id, v.pieces[id].c_str(), (int) v.control[id], (int) v.eog[id]);
    }
    return 0;
}
