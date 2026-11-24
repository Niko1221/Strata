#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include <cstring>
#include <limits>
#include <map>

namespace strata::hy4_flash {

// Stable mapped tensor bytes across GGUF shards. Virtual offsets retain the
// single-file callers' offset arithmetic; they are never passed to file I/O.
// GgufModel owns the mappings and validates shard membership and duplicate names.
class ModelFile {
public:
    explicit ModelFile(const std::string& path) : model_(GgufModel::open(path)) {
        uint64_t base = 0;
        for (std::size_t i = 0; i < model_.size(); ++i) {
            const auto& file = model_.shard(i);
            if (file.file_size() > std::numeric_limits<uint64_t>::max() - base)
                throw std::runtime_error("hy4-flash GGUF shard offsets overflow");
            segments_.push_back({base + file.data_start(), base + file.file_size(), i});
            for (const auto& tensor : file.tensors()) {
                if (!model_.in_bounds(tensor, i))
                    throw std::runtime_error("hy4-flash GGUF unsupported or truncated tensor: " + tensor.name);
                auto view = tensor;
                view.offset += base + file.data_start() - data_start();
                const std::string name=view.name;
                tensors_.emplace(name, std::move(view));
            }
            base += file.file_size();
        }
    }

    const TensorInfo* info(const std::string& name) const {
        const auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    uint64_t data_start() const { return model_.meta().data_start(); }

    const void* at(uint64_t offset, std::size_t bytes) const {
        const auto* segment = locate(offset, bytes);
        if (!segment) return nullptr;
        TensorInfo slice;
        slice.offset = offset - segment->begin;
        return model_.shard(segment->shard).tensor_data(slice);
    }
    bool read_at(uint64_t offset, void* dst, std::size_t bytes) const {
        const void* src = at(offset, bytes);
        if (!src) return false;
        std::memcpy(dst, src, bytes);
        return true;
    }
    bool pread_at(uint64_t offset, void* dst, std::size_t bytes) const {
        const auto* segment = locate(offset, bytes);
        if (!segment) return false;
        const auto& file = model_.shard(segment->shard);
        return file.pread_at(file.data_start() + (offset - segment->begin), dst, bytes);
    }
    // Translate a virtual offset when a shared reader needs a physical file.
    bool physical(uint64_t offset, std::size_t bytes, std::string& path, uint64_t& file_offset) const {
        const auto* segment = locate(offset, bytes);
        if (!segment) return false;
        const auto& file = model_.shard(segment->shard);
        path = file.path();
        file_offset = file.data_start() + offset - segment->begin;
        return true;
    }

private:
    struct Segment { uint64_t begin, end; std::size_t shard; };
    const Segment* locate(uint64_t offset, std::size_t bytes) const {
        for (const auto& segment : segments_)
            if (offset >= segment.begin && offset <= segment.end && bytes <= segment.end - offset)
                return &segment;
        return nullptr;
    }
    GgufModel model_;
    std::vector<Segment> segments_;
    std::map<std::string, TensorInfo> tensors_;
};

} // namespace strata::hy4_flash
