// include/strata/artifact/qwen_tokenizer.hpp - Qwen byte-level BPE tokenizer (vocab.json + merges.txt).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::artifact {

class QwenTokenizer {
public:
    bool load(const std::string& model_dir) {
        init_byte_tables();
        if (!load_vocab(model_dir + "/vocab.json")) return false;
        if (!load_merges(model_dir + "/merges.txt")) return false;
        const std::pair<std::string, int32_t> specials[] = {
            {"<|endoftext|>", 248044},
            {"<|im_start|>", 248045},
            {"<|im_end|>", 248046},
            {"<tool_call>", 248058},
            {"</tool_call>", 248059},
            {"<tool_response>", 248066},
            {"</tool_response>", 248067},
            {"<think>", 248068},
            {"</think>", 248069},
            {"<|think|>", 248068},
            {"<|/think|>", 248069},
        };
        for (const auto& sp : specials) {
            vocab_[sp.first] = sp.second;
            if (id_to_raw_.find(sp.second) == id_to_raw_.end()) {
                id_to_raw_[sp.second] = sp.first;
            }
            specials_.push_back(sp);
        }
        return true;
    }

    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> out;
        size_t pos = 0;
        while (pos < text.size()) {
            size_t next_spec_pos = std::string::npos;
            size_t spec_idx = 0;
            for (size_t i = 0; i < specials_.size(); ++i) {
                size_t f = text.find(specials_[i].first, pos);
                if (f != std::string::npos && f < next_spec_pos) {
                    next_spec_pos = f;
                    spec_idx = i;
                }
            }
            if (next_spec_pos == pos) {
                out.push_back(specials_[spec_idx].second);
                pos += specials_[spec_idx].first.size();
                continue;
            }
            size_t chunk_end = (next_spec_pos == std::string::npos) ? text.size() : next_spec_pos;
            encode_chunk(text.substr(pos, chunk_end - pos), out);
            pos = chunk_end;
        }
        return out;
    }

    std::string decode_token(int32_t id) const {
        auto it = id_to_raw_.find(id);
        if (it != id_to_raw_.end()) return it->second;
        auto bit = id_to_bpe_.find(id);
        if (bit == id_to_bpe_.end()) return "";
        return bpe_str_to_bytes(bit->second);
    }

private:
    static std::string codepoint_to_utf8(int cp) {
        std::string s;
        if (cp <= 0x7F) {
            s.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            s.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        return s;
    }

    void init_byte_tables() {
        std::vector<int> bs;
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        for (size_t i = 0; i < 256; ++i) {
            std::string u = codepoint_to_utf8(cs[i]);
            byte_to_utf8_[bs[i]] = u;
            utf8_to_byte_[u] = static_cast<uint8_t>(bs[i]);
        }
    }

    std::string bpe_str_to_bytes(const std::string& bpe) const {
        std::string res;
        size_t i = 0;
        while (i < bpe.size()) {
            uint8_t c = static_cast<uint8_t>(bpe[i]);
            size_t clen = 1;
            if ((c & 0xE0) == 0xC0) clen = 2;
            else if ((c & 0xF0) == 0xE0) clen = 3;
            else if ((c & 0xF8) == 0xF0) clen = 4;
            std::string ch = bpe.substr(i, clen);
            auto it = utf8_to_byte_.find(ch);
            if (it != utf8_to_byte_.end()) {
                res.push_back(static_cast<char>(it->second));
            } else {
                res += ch;
            }
            i += clen;
        }
        return res;
    }

    static bool parse_json_string(const std::string& src, size_t& pos, std::string& out) {
        out.clear();
        if (pos >= src.size() || src[pos] != '"') return false;
        pos++;
        while (pos < src.size()) {
            char c = src[pos++];
            if (c == '"') return true;
            if (c == '\\' && pos < src.size()) {
                char esc = src[pos++];
                if (esc == '"' || esc == '\\' || esc == '/') out.push_back(esc);
                else if (esc == 'b') out.push_back('\b');
                else if (esc == 'f') out.push_back('\f');
                else if (esc == 'n') out.push_back('\n');
                else if (esc == 'r') out.push_back('\r');
                else if (esc == 't') out.push_back('\t');
                else if (esc == 'u' && pos + 4 <= src.size()) {
                    int cp = std::stoi(src.substr(pos, 4), nullptr, 16);
                    pos += 4;
                    out += codepoint_to_utf8(cp);
                }
            } else {
                out.push_back(c);
            }
        }
        return false;
    }

    bool load_vocab(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        size_t pos = txt.find('{');
        if (pos == std::string::npos) return false;
        pos++;
        std::string key;
        while (pos < txt.size()) {
            while (pos < txt.size() && (txt[pos] == ' ' || txt[pos] == '\n' || txt[pos] == '\r' || txt[pos] == '\t' || txt[pos] == ',')) pos++;
            if (pos >= txt.size() || txt[pos] == '}') break;
            if (!parse_json_string(txt, pos, key)) break;
            while (pos < txt.size() && (txt[pos] == ' ' || txt[pos] == ':')) pos++;
            size_t num_end = pos;
            while (num_end < txt.size() && (txt[num_end] >= '0' && txt[num_end] <= '9')) num_end++;
            int32_t id = std::stoi(txt.substr(pos, num_end - pos));
            vocab_[key] = id;
            id_to_bpe_[id] = key;
            pos = num_end;
        }
        return !vocab_.empty();
    }

    bool load_merges(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        int rank = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::string a = line.substr(0, sp);
            std::string b = line.substr(sp + 1);
            if (!b.empty() && b.back() == '\r') b.pop_back();
            merges_[a + "\x01" + b] = rank++;
        }
        return true;
    }

    static std::vector<std::string> pretokenize(const std::string& s) {
        std::vector<std::string> words;
        size_t i = 0;
        auto is_letter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (static_cast<uint8_t>(c) >= 128); };
        auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
        while (i < s.size()) {
            if (s[i] == ' ' && i + 1 < s.size() && s[i + 1] != ' ' && s[i + 1] != '\n' && s[i + 1] != '\r') {
                size_t start = i++;
                if (is_letter(s[i])) {
                    while (i < s.size() && is_letter(s[i])) i++;
                } else if (is_digit(s[i])) {
                    i++;
                } else {
                    while (i < s.size() && !is_letter(s[i]) && !is_digit(s[i]) && s[i] != ' ' && s[i] != '\n' && s[i] != '\r') i++;
                }
                words.push_back(s.substr(start, i - start));
            } else if (is_letter(s[i])) {
                size_t start = i;
                while (i < s.size() && is_letter(s[i])) i++;
                words.push_back(s.substr(start, i - start));
            } else if (is_digit(s[i])) {
                words.push_back(s.substr(i, 1));
                i++;
            } else if (s[i] == '\n' || s[i] == '\r') {
                size_t start = i;
                while (i < s.size() && (s[i] == '\n' || s[i] == '\r')) i++;
                words.push_back(s.substr(start, i - start));
            } else if (s[i] == ' ') {
                size_t start = i;
                while (i < s.size() && s[i] == ' ' && (i + 1 == s.size() || s[i + 1] == ' ')) i++;
                if (i == start) i++;
                words.push_back(s.substr(start, i - start));
            } else {
                size_t start = i;
                while (i < s.size() && !is_letter(s[i]) && !is_digit(s[i]) && s[i] != ' ' && s[i] != '\n' && s[i] != '\r') i++;
                words.push_back(s.substr(start, i - start));
            }
        }
        return words;
    }

    void encode_chunk(const std::string& chunk, std::vector<int32_t>& out) const {
        for (const std::string& w : pretokenize(chunk)) {
            std::vector<std::string> pieces;
            pieces.reserve(w.size());
            for (unsigned char b : w) {
                pieces.push_back(byte_to_utf8_[b]);
            }
            while (pieces.size() > 1) {
                int best_rank = std::numeric_limits<int>::max();
                size_t best_idx = 0;
                for (size_t i = 0; i + 1 < pieces.size(); ++i) {
                    auto it = merges_.find(pieces[i] + "\x01" + pieces[i + 1]);
                    if (it != merges_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = i;
                    }
                }
                if (best_rank == std::numeric_limits<int>::max()) break;
                pieces[best_idx] += pieces[best_idx + 1];
                pieces.erase(pieces.begin() + static_cast<ptrdiff_t>(best_idx + 1));
            }
            for (const std::string& p : pieces) {
                auto it = vocab_.find(p);
                if (it != vocab_.end()) {
                    out.push_back(it->second);
                }
            }
        }
    }

    std::string byte_to_utf8_[256];
    std::unordered_map<std::string, uint8_t> utf8_to_byte_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> id_to_bpe_;
    std::unordered_map<int32_t, std::string> id_to_raw_;
    std::unordered_map<std::string, int> merges_;
    std::vector<std::pair<std::string, int32_t>> specials_;
};

}  // namespace strata::artifact
