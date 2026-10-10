// include/strata/artifact/exl3_model.hpp - multi-shard EXL3 model accessor (docs/EXL3.md).
//
// An EXL3 model is a set of safetensors shards plus `model.safetensors.index.json` (tensor -> shard).
// Every quantized linear is `prefix.trellis` (I16 [ki,nj,256*K/16]), `prefix.suh`/`prefix.svh` (F16) and a
// `prefix.mul1` / `prefix.mcg` marker.  This resolves a linear across shards and hands back the raw
// pointers the decode kernels take.  Shards are mmap'd lazily and never copied.
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "strata/artifact/safetensors.hpp"

namespace strata {

struct Exl3Linear {
    const uint16_t* trellis = nullptr;
    const uint16_t* suh = nullptr;
    const uint16_t* svh = nullptr;
    int ki = 0, nj = 0, bits = 0;
    int cb = 2;                        // 0=3inst, 1=mcg, 2=mul1
    int in() const { return ki * 16; }
    int out() const { return nj * 16; }
};

namespace exl3_detail {

// Parse the `weight_map` object of a safetensors index: {"tensor.name":"model-0000N-of-0000M.safetensors", ...}.
inline std::map<std::string, std::string> parse_weight_map(const std::string& json) {
    std::map<std::string, std::string> out;
    size_t p = json.find("\"weight_map\"");
    if (p == std::string::npos) throw std::runtime_error("index.json: no weight_map");
    p = json.find('{', p);
    if (p == std::string::npos) throw std::runtime_error("index.json: malformed weight_map");
    ++p;
    auto skip_ws = [&]() { while (p < json.size() && (json[p] == ' ' || json[p] == '\n' || json[p] == '\r' || json[p] == '\t')) ++p; };
    auto str = [&]() -> std::string {
        if (p >= json.size() || json[p] != '"') throw std::runtime_error("index.json: expected string");
        ++p;
        std::string s;
        while (p < json.size() && json[p] != '"') {
            if (json[p] == '\\' && p + 1 < json.size()) { ++p; }
            s.push_back(json[p++]);
        }
        ++p;
        return s;
    };
    skip_ws();
    if (json[p] == '}') return out;
    for (;;) {
        skip_ws();
        std::string key = str();
        skip_ws(); ++p;                        // ':'
        skip_ws();
        std::string val = str();
        out.emplace(std::move(key), std::move(val));
        skip_ws();
        if (json[p] == ',') { ++p; continue; }
        if (json[p] == '}') break;
        throw std::runtime_error("index.json: malformed weight_map entry");
    }
    return out;
}

}  // namespace exl3_detail

class Exl3Model {
public:
    explicit Exl3Model(const std::string& dir) : dir_(dir) {
        std::string path = dir + "/model.safetensors.index.json";
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("exl3: cannot open " + path);
        std::ostringstream ss;
        ss << in.rdbuf();
        weight_map_ = exl3_detail::parse_weight_map(ss.str());
    }

    bool has(const std::string& base) const { return weight_map_.count(base + ".trellis") != 0; }
    size_t tensor_count() const { return weight_map_.size(); }

    Exl3Linear linear(const std::string& base) const {
        auto tr = tensor(base + ".trellis");
        auto suh = tensor(base + ".suh");
        auto svh = tensor(base + ".svh");
        Exl3Linear L;
        L.trellis = (const uint16_t*)data(*tr);
        L.suh = (const uint16_t*)data(*suh);
        L.svh = (const uint16_t*)data(*svh);
        L.ki = (int)tr->shape[0];
        L.nj = (int)tr->shape[1];
        L.bits = (int)(tr->shape[2] * 16 / 256);
        L.cb = weight_map_.count(base + ".mul1") ? 2 : weight_map_.count(base + ".mcg") ? 1 : 0;
        return L;
    }

private:
    const SafetensorsFile& shard(const std::string& name) const {
        auto it = shards_.find(name);
        if (it != shards_.end()) return *it->second;
        auto f = std::make_unique<SafetensorsFile>(dir_ + "/" + name);
        auto* raw = f.get();
        shards_.emplace(name, std::move(f));
        return *raw;
    }
    const StTensor* tensor(const std::string& name) const {
        auto it = weight_map_.find(name);
        if (it == weight_map_.end()) throw std::runtime_error("exl3: no tensor " + name);
        const StTensor* t = shard(it->second).find(name);
        if (!t) throw std::runtime_error("exl3: tensor " + name + " not in " + it->second);
        return t;
    }
    const uint8_t* data(const StTensor& t) const {
        auto it = weight_map_.find(t.name);
        return shard(it->second).data(t);
    }

    std::string dir_;
    std::map<std::string, std::string> weight_map_;
    mutable std::map<std::string, std::unique_ptr<SafetensorsFile>> shards_;
};

}  // namespace strata
