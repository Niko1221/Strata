#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::qwen38 {
// GPU weights are read through bounded pageable staging, never touched through
// the long-lived GGUF mmap. That mapping is retained for metadata/embedding only.
class UploadReader {
    std::ifstream input_;
    uint64_t size_;
    std::vector<uint8_t> staging_;
public:
    static constexpr size_t chunk_bytes=16*1024*1024;
    UploadReader(const std::string& path,uint64_t expected_size):input_(path,std::ios::binary),size_(expected_size),staging_(chunk_bytes){
        if(!input_)throw std::runtime_error("cannot open bounded weight reader");
        input_.seekg(0,std::ios::end);auto n=input_.tellg();
        if(n<0||uint64_t(n)!=size_)throw std::runtime_error("weight file size changed after admission");
    }
    size_t staging_bytes()const{return staging_.size();}
    // consume must finish using the supplied span before returning. A failed
    // read never delivers a partial chunk. Earlier delivered chunks may exist;
    // constructor RAII discards incomplete GPU weights on any exception.
    template<class Before,class Consume>void read(uint64_t offset,uint64_t bytes,Before before,Consume consume){
        if(offset>size_||bytes>size_-offset||offset>uint64_t(std::numeric_limits<std::streamoff>::max()))
            throw std::runtime_error("weight read outside validated file extent");
        input_.clear();input_.seekg(std::streamoff(offset));if(!input_)throw std::runtime_error("weight seek failed");
        for(uint64_t done=0;done<bytes;){
            before();size_t n=size_t(std::min<uint64_t>(bytes-done,staging_.size()));
            input_.read(reinterpret_cast<char*>(staging_.data()),std::streamsize(n));
            if(input_.gcount()!=std::streamsize(n)||!input_)throw std::runtime_error("short weight read");
            consume(done,staging_.data(),n);done+=n;
        }
    }
};
}
