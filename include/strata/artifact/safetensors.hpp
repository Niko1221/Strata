// include/strata/artifact/safetensors.hpp - safetensors reader for EXL3 models (docs/EXL3.md).
//
// The EXL3 format lives in safetensors, not GGUF, so this is the container the engine needs next to
// gguf_reader.  The format is: 8-byte little-endian header length N, N bytes of JSON, then the tensor
// data buffer; every tensor records `dtype`, `shape` and `data_offsets` (relative to the data buffer).
//
// LIKE gguf_reader: mmap, no ggml dependency, and a small header parser.  The JSON here is the
// safetensors header only (a flat object of tensor descriptors plus an optional __metadata__), so a
// full JSON library would be dead weight; the parser below handles exactly that grammar.
//
// Scope: POSIX (mmap) for now.  A Windows path (CreateFileMapping) is a follow-up; nothing in the
// engine includes this header yet.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata {

struct StTensor {
    std::string name;
    std::string dtype;                 // "I16", "F16", "F32", "I32", ...
    std::vector<int64_t> shape;
    size_t begin = 0;                  // byte offsets into the data buffer
    size_t end = 0;
    size_t nbytes() const { return end - begin; }
    size_t nelem() const {
        size_t n = 1;
        for (int64_t s : shape) n *= (size_t)s;
        return n;
    }
};

namespace safetensors_detail {

// A minimal recursive-descent parser for the safetensors header grammar.  It reads tensor
// descriptors and skips everything else (notably __metadata__), so it never has to build a general
// JSON value tree for a multi-megabyte header.
class HeaderParser {
public:
    HeaderParser(const char* p, const char* end) : p_(p), end_(end) {}

    std::map<std::string, StTensor> parse() {
        std::map<std::string, StTensor> out;
        ws();
        expect('{');
        ws();
        if (peek() == '}') { ++p_; return out; }
        for (;;) {
            std::string key = string();
            ws(); expect(':'); ws();
            if (key == "__metadata__") {
                skip_value();
            } else {
                out[key] = descriptor(key);
            }
            ws();
            char c = peek();
            if (c == ',') { ++p_; ws(); continue; }
            if (c == '}') { ++p_; break; }
            throw std::runtime_error("safetensors: expected ',' or '}' in header");
        }
        return out;
    }

private:
    const char* p_;
    const char* end_;

    [[noreturn]] void fail(const char* what) const { throw std::runtime_error(std::string("safetensors header: ") + what); }
    char peek() const { if (p_ >= end_) fail("unexpected end"); return *p_; }
    void expect(char c) { if (peek() != c) fail("unexpected char"); ++p_; }
    void ws() { while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_; }

    std::string string() {
        expect('"');
        std::string s;
        while (p_ < end_ && *p_ != '"') {
            char c = *p_++;
            if (c == '\\') {
                if (p_ >= end_) fail("bad escape");
                char e = *p_++;
                switch (e) {
                    case 'n': s.push_back('\n'); break;
                    case 't': s.push_back('\t'); break;
                    case '"': s.push_back('"'); break;
                    case '\\': s.push_back('\\'); break;
                    case '/': s.push_back('/'); break;
                    case 'u': {                       // keep it simple: decode BMP code point to UTF-8
                        unsigned cp = 0;
                        for (int i = 0; i < 4; ++i) cp = (cp << 4) | hex();
                        if (cp < 0x80) s.push_back((char)cp);
                        else if (cp < 0x800) { s.push_back((char)(0xC0 | (cp >> 6))); s.push_back((char)(0x80 | (cp & 0x3F))); }
                        else { s.push_back((char)(0xE0 | (cp >> 12))); s.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); s.push_back((char)(0x80 | (cp & 0x3F))); }
                        break;
                    }
                    default: fail("bad escape");
                }
            } else {
                s.push_back(c);
            }
        }
        expect('"');
        return s;
    }

    unsigned hex() {
        if (p_ >= end_) fail("bad \\u");
        char c = *p_++;
        if (c >= '0' && c <= '9') return (unsigned)(c - '0');
        if (c >= 'a' && c <= 'f') return (unsigned)(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return (unsigned)(c - 'A' + 10);
        fail("bad hex");
    }

    int64_t number() {
        bool neg = false;
        if (peek() == '-') { neg = true; ++p_; }
        int64_t v = 0;
        bool any = false;
        while (p_ < end_ && *p_ >= '0' && *p_ <= '9') { v = v * 10 + (*p_ - '0'); ++p_; any = true; }
        if (!any) fail("bad number");
        return neg ? -v : v;
    }

    std::vector<int64_t> int_array() {
        expect('[');
        ws();
        std::vector<int64_t> v;
        if (peek() == ']') { ++p_; return v; }
        for (;;) {
            v.push_back(number());
            ws();
            char c = peek();
            if (c == ',') { ++p_; ws(); continue; }
            if (c == ']') { ++p_; break; }
            fail("bad array");
        }
        (void)v;
        return v;
    }

    StTensor descriptor(const std::string& name) {
        StTensor t;
        t.name = name;
        ws(); expect('{'); ws();
        bool first = true;
        for (;;) {
            if (peek() == '}') { ++p_; break; }
            if (!first) { expect(','); ws(); }
            first = false;
            std::string k = string();
            ws(); expect(':'); ws();
            if (k == "dtype") t.dtype = string();
            else if (k == "shape") t.shape = int_array();
            else if (k == "data_offsets") { auto o = int_array(); if (o.size() != 2) fail("data_offsets"); t.begin = (size_t)o[0]; t.end = (size_t)o[1]; }
            else skip_value();
            ws();
        }
        return t;
    }

    void skip_value() {
        ws();
        char c = peek();
        if (c == '"') { string(); return; }
        if (c == '{') {
            int depth = 0;
            do {
                char d = *p_++;
                if (d == '"') { --p_; string(); continue; }
                if (d == '{') ++depth;
                else if (d == '}') --depth;
                if (p_ >= end_) fail("unterminated object");
            } while (depth > 0);
            return;
        }
        if (c == '[') {
            int depth = 0;
            do {
                char d = *p_++;
                if (d == '"') { --p_; string(); continue; }
                if (d == '[') ++depth;
                else if (d == ']') --depth;
                if (p_ >= end_) fail("unterminated array");
            } while (depth > 0);
            return;
        }
        while (p_ < end_ && (std::strchr(" \t\r\n,}]", *p_) == nullptr)) ++p_;
    }
};

}  // namespace safetensors_detail

// One safetensors file, mmap'd.  Tensor data is a raw pointer into the mapping (zero copy).
class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path) : path_(path) {
#if defined(_WIN32)
        throw std::runtime_error("safetensors: POSIX mmap only (Windows path not implemented yet)");
#else
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("safetensors: cannot open " + path);
        struct stat sb;
        if (::fstat(fd, &sb) != 0) { ::close(fd); throw std::runtime_error("safetensors: fstat failed"); }
        size_ = (size_t)sb.st_size;
        base_ = (const uint8_t*)::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (base_ == MAP_FAILED) { base_ = nullptr; throw std::runtime_error("safetensors: mmap failed"); }
#endif
        if (size_ < 8) throw std::runtime_error("safetensors: file too small");
        uint64_t n = 0;
        std::memcpy(&n, base_, 8);
        if (n == 0 || 8 + n > size_) throw std::runtime_error("safetensors: bad header length");
        header_len_ = (size_t)n;
        data_offset_ = 8 + header_len_;
        const char* hp = reinterpret_cast<const char*>(base_ + 8);
        tensors_ = safetensors_detail::HeaderParser(hp, hp + header_len_).parse();
    }

    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    ~SafetensorsFile() {
#if !defined(_WIN32)
        if (base_) ::munmap((void*)base_, size_);
#endif
    }

    const StTensor* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    const uint8_t* data(const StTensor& t) const { return base_ + data_offset_ + t.begin; }
    const std::map<std::string, StTensor>& tensors() const { return tensors_; }
    const std::string& path() const { return path_; }
    size_t size() const { return size_; }

private:
    std::string path_;
    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
    size_t header_len_ = 0;
    size_t data_offset_ = 0;
    std::map<std::string, StTensor> tensors_;
};

}  // namespace strata
