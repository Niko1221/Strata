#include "strata/core/grammar.hpp"

#include <picojson.h>
#include <xgrammar/tokenizer_info.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace strata::grammar {
namespace {
picojson::value read_json(const std::filesystem::path& path, size_t limit) {
    const auto size = std::filesystem::file_size(path);
    if (size > limit) throw std::runtime_error("grammar tokenizer file exceeds its size limit");
    std::ifstream in(path, std::ios::binary);
    std::string data((size_t) size, '\0');
    if (!in.read(data.data(), (std::streamsize) data.size()))
        throw std::runtime_error("could not read grammar tokenizer artifact");
    picojson::value value;
    const std::string error = picojson::parse(value, data);
    if (!error.empty()) throw std::runtime_error("invalid grammar tokenizer artifact: " + error);
    return value;
}

std::string vocabulary_identity(const std::vector<std::string>& bytes, const std::vector<int32_t>& stops) {
    // Diagnostic only. Cache keys compare full source and immutable vocabulary
    // ownership; checkpoints retain the exact compiled object, not this hash.
    uint64_t h = 14695981039346656037ull;
    auto octet = [&](uint8_t b) { h = (h ^ b) * 1099511628211ull; };
    auto number = [&](uint64_t n) { for (int i = 0; i < 8; ++i) octet((uint8_t) (n >> (i * 8))); };
    number(bytes.size());
    for (const auto& token : bytes) {
        number(token.size());
        for (unsigned char b : token) octet(b);
    }
    number(stops.size());
    for (int32_t id : stops) number((uint64_t) id);
    std::ostringstream out;
    out << "bytes-v1-fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << h;
    return out.str();
}
} // namespace

std::shared_ptr<const Vocabulary> Vocabulary::from_bytes(std::vector<std::string> bytes,
                                                        std::vector<int32_t> stops, ProtocolTokens protocol) {
    if (bytes.empty() || bytes.size() > 300000 || stops.empty() || stops.size() > 16)
        throw std::runtime_error("unsupported grammar vocabulary or stop-token count");
    std::sort(stops.begin(), stops.end());
    if (std::adjacent_find(stops.begin(), stops.end()) != stops.end())
        throw std::runtime_error("duplicate grammar stop-token ID");
    for (int32_t id : stops) {
        if (id < 0 || (size_t) id >= bytes.size()) throw std::runtime_error("grammar stop-token ID out of range");
        bytes[(size_t) id].clear();
    }
    size_t total = 0;
    for (const auto& token : bytes) {
        total += token.size();
        if (token.size() > 4096 || total > 32 * 1024 * 1024)
            throw std::runtime_error("grammar token byte table exceeds its resource limit");
    }
    auto out = std::make_shared<Vocabulary>();
    std::vector<int32_t> controls;
    for (auto id : {protocol.think_end, protocol.call_start, protocol.call_end}) {
        if (id == -1) continue;
        if (id < 0 || (size_t) id >= bytes.size() || !bytes[id].empty() ||
            std::find(stops.begin(), stops.end(), id) != stops.end() ||
            std::find(controls.begin(), controls.end(), id) != controls.end())
            throw std::runtime_error("invalid grammar protocol token IDs");
        controls.push_back(id);
    }
    out->protocol = protocol;
    out->text_mask.resize((bytes.size() + 31) / 32);
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (bytes[i].empty()) continue;
        out->text_mask[i / 32] |= (int32_t) (1u << (i % 32));
        if (bytes[i].find_first_not_of('\n') == std::string::npos) out->newline_ids.push_back((int32_t) i);
    }
    out->identity = vocabulary_identity(bytes, stops);
    out->bytes = std::move(bytes);
    out->stop_ids = std::move(stops);
    return out;
}

std::shared_ptr<const Vocabulary> Vocabulary::from_pack(const std::filesystem::path& path,
                                                       std::vector<int32_t> stops) {
    // These are setup's existing tokenizer artifacts, not a second tokenizer.
    // XGrammar decodes the GPT-2 byte alphabet; prompt BPE stays in Strata.
    const auto config = read_json(path / "tokenizer.json", 64 * 1024);
    if (!config.is<picojson::object>()) throw std::runtime_error("grammar tokenizer metadata must be an object");
    const auto& meta = config.get<picojson::object>();
    auto string_is = [&](const char* key, const char* wanted) {
        const auto it = meta.find(key);
        return it != meta.end() && it->second.is<std::string>() && it->second.get<std::string>() == wanted;
    };
    if (!string_is("model", "gpt2") || !string_is("pre", "qwen35"))
        throw std::runtime_error("native grammar currently supports the gpt2/qwen35 tokenizer artifacts only");
    if (auto it = meta.find("add_prefix_space"); it != meta.end() &&
        (!it->second.is<bool>() || it->second.get<bool>()))
        throw std::runtime_error("native grammar does not support add_prefix_space");

    const auto v = read_json(path / "vocab.json", 32 * 1024 * 1024);
    const auto t = read_json(path / "token_type.json", 4 * 1024 * 1024);
    if (!v.is<picojson::object>() || !t.is<picojson::array>())
        throw std::runtime_error("invalid grammar vocabulary/token-type artifacts");
    const auto& vocab = v.get<picojson::object>();
    const auto& types = t.get<picojson::array>();
    if (vocab.size() != types.size() || vocab.empty() || vocab.size() > 300000)
        throw std::runtime_error("grammar vocabulary and token-type sizes differ or exceed the limit");
    std::vector<std::string> encoded(vocab.size());
    std::vector<bool> seen(vocab.size());
    ProtocolTokens protocol;
    for (const auto& [token, id_value] : vocab) {
        if (!id_value.is<double>()) throw std::runtime_error("grammar token ID must be an integer");
        const double raw = id_value.get<double>();
        if (!std::isfinite(raw) || std::floor(raw) != raw || raw < 0 || raw >= vocab.size())
            throw std::runtime_error("grammar token ID is out of range");
        const size_t id = (size_t) raw;
        if (seen[id] || !types[id].is<double>()) throw std::runtime_error("duplicate ID or invalid token type");
        seen[id] = true;
        const double type = types[id].get<double>();
        if (type < 1 || type > 5 || std::floor(type) != type)
            throw std::runtime_error("unsupported GPT-2 token type for native grammar");
        if (type == 1) encoded[id] = token;
        if (type == 4) {
            if (token == "</think>") protocol.think_end = (int32_t) id;
            if (token == "<tool_call>") protocol.call_start = (int32_t) id;
            if (token == "</tool_call>") protocol.call_end = (int32_t) id;
        }
    }
    const xgrammar::TokenizerInfo info(encoded, xgrammar::VocabType::BYTE_LEVEL,
                                       (int) encoded.size(), stops, false);
    return from_bytes(info.GetDecodedVocab(), std::move(stops), protocol);
}

} // namespace strata::grammar
