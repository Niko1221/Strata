#include "dsv4/gguf_header.hpp"

#include <cstdio>
#include <cstring>
#include <set>

namespace dsv4 {
namespace {

struct TypeRow { uint32_t id; const char* name; int block; int bytes; };
const TypeRow kTypes[] = {
    {0, "F32", 1, 4},     {1, "F16", 1, 2},       {2, "Q4_0", 32, 18},    {3, "Q4_1", 32, 20},
    {6, "Q5_0", 32, 22},  {7, "Q5_1", 32, 24},    {8, "Q8_0", 32, 34},    {10, "Q2_K", 256, 84},
    {11, "Q3_K", 256, 110}, {12, "Q4_K", 256, 144}, {13, "Q5_K", 256, 176}, {14, "Q6_K", 256, 210},
    {15, "Q8_K", 256, 292}, {16, "IQ2_XXS", 256, 66}, {17, "IQ2_XS", 256, 74}, {18, "IQ3_XXS", 256, 98},
    {19, "IQ1_S", 256, 50}, {20, "IQ4_NL", 32, 18}, {21, "IQ3_S", 256, 110}, {22, "IQ2_S", 256, 82},
    {23, "IQ4_XS", 256, 136}, {24, "I8", 1, 1},   {25, "I16", 1, 2},      {26, "I32", 1, 4},
    {27, "I64", 1, 8},    {28, "F64", 1, 8},      {29, "IQ1_M", 256, 56}, {30, "BF16", 1, 2},
    {39, "MXFP4", 32, 17},
};

struct Rd {
    FILE* f = nullptr;
    std::string err;
    bool raw(void* p, size_t n) {
        if (n == 0) return true;
        if (std::fread(p, 1, n, f) != n) { err = "truncated GGUF header"; return false; }
        return true;
    }
    template <class T> bool pod(T& v) { return raw(&v, sizeof(T)); }
    bool str(std::string& s) {
        uint64_t n = 0;
        if (!pod(n)) return false;
        if (n > (1ULL << 28)) { err = "implausible string length"; return false; }
        s.resize((size_t) n);
        return raw(&s[0], (size_t) n);
    }
};

size_t scalar_size(uint32_t t) {
    switch (t) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0;
    }
}

bool read_scalar(Rd& r, uint32_t t, double& d, int64_t& i, uint64_t& u) {
    switch (t) {
        case GGUF_U8: case GGUF_BOOL: { uint8_t v; if (!r.pod(v)) return false; u = v; i = v; d = v; return true; }
        case GGUF_I8: { int8_t v; if (!r.pod(v)) return false; i = v; u = (uint64_t) i; d = v; return true; }
        case GGUF_U16: { uint16_t v; if (!r.pod(v)) return false; u = v; i = v; d = v; return true; }
        case GGUF_I16: { int16_t v; if (!r.pod(v)) return false; i = v; u = (uint64_t) i; d = v; return true; }
        case GGUF_U32: { uint32_t v; if (!r.pod(v)) return false; u = v; i = v; d = v; return true; }
        case GGUF_I32: { int32_t v; if (!r.pod(v)) return false; i = v; u = (uint64_t) i; d = v; return true; }
        case GGUF_F32: { float v; if (!r.pod(v)) return false; d = v; i = (int64_t) v; u = (uint64_t) i; return true; }
        case GGUF_U64: { uint64_t v; if (!r.pod(v)) return false; u = v; i = (int64_t) v; d = (double) v; return true; }
        case GGUF_I64: { int64_t v; if (!r.pod(v)) return false; i = v; u = (uint64_t) v; d = (double) v; return true; }
        case GGUF_F64: { double v; if (!r.pod(v)) return false; d = v; i = (int64_t) v; u = (uint64_t) i; return true; }
        default: r.err = "not a scalar type"; return false;
    }
}

bool read_value(Rd& r, uint32_t t, GgufValue& v) {
    v.type = t;
    if (scalar_size(t)) return read_scalar(r, t, v.f, v.i, v.u);
    if (t == GGUF_STR) return r.str(v.s);
    if (t != GGUF_ARR) { r.err = "unknown kv value type " + std::to_string(t); return false; }
    if (!r.pod(v.elem_type) || !r.pod(v.arr_len)) return false;
    if (v.arr_len > (1ULL << 28)) { r.err = "implausible array length"; return false; }
    if (scalar_size(v.elem_type)) {
        v.nums.reserve((size_t) (v.arr_len < kMaxStoredNums ? v.arr_len : kMaxStoredNums));
        for (uint64_t k = 0; k < v.arr_len; ++k) {
            double d; int64_t i; uint64_t u;
            if (!read_scalar(r, v.elem_type, d, i, u)) return false;
            if (k < kMaxStoredNums) v.nums.push_back(d);
        }
        if (v.arr_len > kMaxStoredNums) v.nums.clear();  // a partial copy is worse than none
        return true;
    }
    if (v.elem_type == GGUF_STR) {
        std::string tmp;
        for (uint64_t k = 0; k < v.arr_len; ++k) {
            if (!r.str(tmp)) return false;
            if (k < kMaxStoredStrs) v.strs.push_back(tmp);
        }
        return true;
    }
    r.err = "unsupported array element type " + std::to_string(v.elem_type);
    return false;
}

}  // namespace

bool ggml_type_info(uint32_t type, const char** name, int* block_elems, int* block_bytes) {
    for (const TypeRow& row : kTypes) {
        if (row.id == type) {
            if (name) *name = row.name;
            if (block_elems) *block_elems = row.block;
            if (block_bytes) *block_bytes = row.bytes;
            return true;
        }
    }
    return false;
}

std::string ggml_type_str(uint32_t type) {
    const char* n = nullptr;
    if (ggml_type_info(type, &n, nullptr, nullptr)) return n;
    return "type#" + std::to_string(type);
}

bool gguf_read_header(const std::vector<std::string>& paths, GgufHeader& out, std::string& err) {
    out = GgufHeader();
    std::set<std::string> seen;
    for (size_t si = 0; si < paths.size(); ++si) {
        Rd r;
        r.f = std::fopen(paths[si].c_str(), "rb");
        if (!r.f) { err = "cannot open " + paths[si]; return false; }
        struct Closer { FILE* f; ~Closer() { std::fclose(f); } } closer{r.f};
        char magic[4];
        if (!r.raw(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) { err = paths[si] + ": not a GGUF file"; return false; }
        uint32_t ver = 0; uint64_t n_t = 0, n_kv = 0;
        if (!r.pod(ver) || !r.pod(n_t) || !r.pod(n_kv)) { err = paths[si] + ": " + r.err; return false; }
        if (ver < 2 || ver > 3) { err = paths[si] + ": unsupported GGUF version " + std::to_string(ver); return false; }
        if (n_t > (1ULL << 24) || n_kv > (1ULL << 24)) { err = paths[si] + ": implausible counts"; return false; }
        if (si == 0) out.version = ver;
        out.files.push_back(paths[si]);
        for (uint64_t k = 0; k < n_kv; ++k) {
            std::string key; uint32_t t = 0; GgufValue v;
            if (!r.str(key) || !r.pod(t) || !read_value(r, t, v)) { err = paths[si] + ": " + r.err; return false; }
            out.kv.insert(std::make_pair(key, std::move(v)));  // first shard wins
        }
        uint64_t alignment = 32;
        if (const GgufValue* v = out.find("general.alignment")) if (v->u > 0) alignment = v->u;
        std::vector<uint64_t> shard_tensor_idx;
        for (uint64_t k = 0; k < n_t; ++k) {
            GgufTensor t; uint32_t nd = 0;
            if (!r.str(t.name) || !r.pod(nd)) { err = paths[si] + ": " + r.err; return false; }
            if (nd == 0 || nd > 8) { err = paths[si] + ": bad n_dims for " + t.name; return false; }
            t.dims.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) if (!r.pod(t.dims[d])) { err = paths[si] + ": " + r.err; return false; }
            if (!r.pod(t.type) || !r.pod(t.offset)) { err = paths[si] + ": " + r.err; return false; }
            shard_tensor_idx.push_back(out.tensors.size());   // abs_offset is filled once data_start is known
            const char* nm = nullptr; int be = 0, bb = 0;
            t.known_type = ggml_type_info(t.type, &nm, &be, &bb);
            if (t.known_type) {
                unsigned __int128 elems = 1;
                for (uint64_t d : t.dims) elems *= d;
                if (elems % (unsigned) be != 0) t.known_type = false;
                else t.nbytes = (uint64_t) (elems / (unsigned) be) * (uint64_t) bb;
            }
            t.shard = (int) si;
            if (!seen.insert(t.name).second) { err = "duplicate tensor across shards: " + t.name; return false; }
            out.tensors.push_back(std::move(t));
        }
        // Tensor data starts at the next multiple of general.alignment after the info section.
        const long here = std::ftell(r.f);
        if (here < 0) { err = paths[si] + ": cannot locate the tensor-data section"; return false; }
        const uint64_t data_start = ((uint64_t) here + alignment - 1) / alignment * alignment;
        out.data_start.push_back(data_start);
        for (uint64_t idx : shard_tensor_idx) out.tensors[idx].abs_offset = data_start + out.tensors[idx].offset;
    }
    return true;
}

}  // namespace dsv4
