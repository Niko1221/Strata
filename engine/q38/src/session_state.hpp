#pragma once
#include "kv.hpp"
namespace strata::qwen38 {
struct RecurrentState {Buffer conv,gdn;};
// Shared production whole-session reset, also exercised by numerical fixtures.
inline void reset_session(std::vector<RecurrentState>& states,KvCache& kv,Buffer& error,int& position,cudaStream_t stream){
    ck(cudaStreamSynchronize(stream));
    for(auto& state:states)if(state.conv.data){state.conv.zero(stream);state.gdn.zero(stream);}
    kv.reset();error.zero(stream);ck(cudaStreamSynchronize(stream));position=0;
}
}
