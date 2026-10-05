#include "strata/core/prefix_store.hpp"
#include "strata/kernels/kv_q4.hpp"
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <string>
#include <limits>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct Fixture {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device, host;
    std::array<void*,5> sources{};
    std::array<size_t,5> sizes{};

    template<class T> void alloc(T*& p, size_t n, bool pinned = false) {
        if (!n) return;
        void* raw = nullptr;
        if (pinned) {
            cuda_check(cudaHostAlloc(&raw, n, cudaHostAllocMapped));
            host.push_back(raw);
            void* mapped = nullptr;
            cuda_check(cudaHostGetDevicePointer(&mapped, raw, 0));
            p = static_cast<T*>(mapped);
        } else {
            cuda_check(cudaMalloc(&raw, n)); device.push_back(raw); p = static_cast<T*>(raw);
        }
    }
    Fixture(int fmt, int mode) {
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt==kKvInt8; st.kv_q4 = fmt==kKvQ4; st.kv_hybrid = fmt==3;
        st.n_pages = 24; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = mode==2 ? 2 : st.max_cells/4+2;
        const size_t per = fmt==kKvQ4 ? kv_q4_bytes_per_head((int)g.head_dim) : g.head_dim*((st.kv_int8 || st.kv_hybrid) ? 1:2);
        const size_t rows = st.max_cells*g.n_head_kv, slot_rows = st.n_slots*4*g.n_head_kv;
        sizes = {rows*per, rows*per, fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0,
                 fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0, (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
        if (fmt==3) {
            sizes = {rows*per, rows*kv_q4_bytes_per_head((int)g.head_dim), rows*(g.head_dim/64)*2, 0,
                     (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
            alloc(st.k_q,sizes[0]); alloc(st.v_q4,sizes[1]); alloc(st.k_scale,sizes[2]);
            sources[0]=st.k_q; sources[1]=st.v_q4; sources[2]=st.k_scale;
        } else if (fmt==kKvQ4) {
            alloc(st.k_q4,slot_rows*per); alloc(st.v_q4,slot_rows*per);
            if (mode) {alloc(st.host.k_q4,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);}
            sources[0]=mode?st.host.k_q4:st.k_q4; sources[1]=mode?st.host.v_q4:st.v_q4;
        } else if (fmt==kKvInt8) {
            alloc(st.k_q,slot_rows*per); alloc(st.v_q,slot_rows*per);
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2); alloc(st.v_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true); alloc(st.host.v_scale,sizes[3],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q:st.v_q;
            sources[2]=mode?st.host.k_scale:st.k_scale; sources[3]=mode?st.host.v_scale:st.v_scale;
        } else {
            alloc(st.k_pool,slot_rows*per); alloc(st.v_pool,slot_rows*per);
            if (mode) {alloc(st.host.k_pool,sizes[0],true); alloc(st.host.v_pool,sizes[1],true);}
            sources[0]=mode?st.host.k_pool:st.k_pool; sources[1]=mode?st.host.v_pool:st.v_pool;
        }
        alloc(st.idx_pooled,sizes[4]); sources[4]=st.idx_pooled;
        alloc(st.page_table,st.n_pages*4);
        if (mode==1) {
            auto& m=st.map;
            m.page_table=st.page_table; m.n_blocks=st.n_pages; m.n_slots=st.n_slots;
            alloc(m.slot_block,st.n_slots*4); alloc(m.slot_stamp,st.n_slots*4); alloc(m.slot_ref,st.n_slots*4);
            alloc(m.miss_block,st.n_slots*4); alloc(m.miss_slot,st.n_slots*4); alloc(m.ctl,kKvCtlInts*4);
        } else if (mode==2) {
            kv_ring_table(st.page_table,st.n_pages,st.n_slots,nullptr);
        }
    }
    void fill(uint8_t salt) {
        for (size_t i=0;i<sources.size();++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j=0;j<data.size();++j) data[j]=(uint8_t)(salt+i*31+j*7+j/257);
            if (!data.empty()) cuda_check(cudaMemcpy(sources[i],data.data(),data.size(),cudaMemcpyDefault));
        }
    }
    void fill_after(uint8_t salt, int64_t first_dirty) {
        for (size_t i=0;i<sources.size();++i) {
            const size_t offset = i == 4 ? size_t(first_dirty/4)*g.idx_key_dim*4
                                         : (sizes[i]/size_t(state.max_cells))*size_t((first_dirty/4)*4);
            if (sizes[i] > offset)
                cuda_check(cudaMemset(static_cast<uint8_t*>(sources[i])+offset, salt, sizes[i]-offset));
        }
    }
    ~Fixture() { for (void* p:device) cudaFree(p); for (void* p:host) cudaFreeHost(p); }
};
bool equal(const ConversationKv& a,const ConversationKv& b) {
    return a.k==b.k && a.v==b.v && a.k_scale==b.k_scale && a.v_scale==b.v_scale && a.pooled==b.pooled;
}
bool same_state(const SavedConversation& a,const SavedConversation& b) {
    if (a.kv.size()!=b.kv.size() || a.live.ids!=b.live.ids) return false;
    for (size_t i=0;i<a.kv.size();++i) if (!equal(a.kv[i],b.kv[i])) return false;
    return a.live.gdn==b.live.gdn && a.live.ple==b.live.ple && a.live.tails==b.live.tails &&
           a.live.dead==b.live.dead && a.live.block_pos==b.live.block_pos;
}

// docs/DETAILS.md: a prefix saved from a session that holds more than it comes back byte for byte by every
// path (streamed from the file, read into RAM, from RAM), the index survives a restart, and files of another
// identity, over the disk budget or cut short are dropped.
void prefix_session(int fmt, int mode, const std::string& dir) {
    namespace fs = std::filesystem;
    fs::remove_all(dir);
    Fixture main(fmt,mode), draft(fmt==3?kKvInt8:fmt,2);
    auto& g=main.g;
    g.n_layers=4; g.n_expert=256;
    g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    SessionState ss;
    ss.max_cells=96; ss.qsa_states=&main.state;
    ss.layer_hi=g.n_layers; ss.gdn_alloc=g.n_gdn_layers(); ss.qsa_alloc=g.n_qsa_layers();
    std::string err,log,source;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g,sizes,err),"geometry sizes");
    main.alloc(ss.gdn_state,sizes.gdn); main.alloc(ss.ple_hist,sizes.ple);
    main.alloc(main.state.idx_tail,sizes.tail); main.alloc(main.state.idx_dead,sizes.dead);
    main.alloc(main.state.idx_block_pos,sizes.block_pos);
    std::vector<int32_t> ids(65);
    for (size_t i=0;i<ids.size();++i) ids[i]=(int32_t)i+1;
    auto fill=[&](uint8_t salt) {
        main.fill(salt); draft.fill(salt);
        for (const auto& [p,n] : std::vector<std::pair<void*,size_t>>{
                 {ss.gdn_state,sizes.gdn},{ss.ple_hist,sizes.ple},{main.state.idx_tail,sizes.tail},
                 {main.state.idx_dead,sizes.dead},{main.state.idx_block_pos,sizes.block_pos}})
            cuda_check(cudaMemset(p,salt,n));
        cuda_check(cudaDeviceSynchronize());
    };
    const std::vector<ConversationImageKey> none;
    const std::vector<ConversationCheckpoint> no_checks;
    // the state at a root of `L` tokens, as a whole-session image: what a restore must reproduce
    auto at_root=[&](uint8_t salt,int64_t L,ConversationCheckpoint& root,SavedConversation& ref) {
        fill(salt);
        root={}; root.ids.assign(ids.begin(),ids.begin()+L);
        check(conversation_checkpoint_save(root,ss,g,err),"save root checkpoint");
        check(conversation_checkpoint_restore(root,ss,g,err),"root spare row");
        const ConversationView view{root.ids,none,no_checks,true};
        check(conversation_snapshot_save(ref,view,ss,g,draft.state,err),"reference image");
        fill(salt);   // the session goes on past the root
    };
    auto restored=[&](const SavedConversation& ref) {
        SavedConversation now;
        const ConversationView view{ref.live.ids,none,no_checks,true};
        cuda_check(cudaDeviceSynchronize());
        check(conversation_snapshot_save(now,view,ss,g,draft.state,err),"read back");
        return same_state(now,ref);
    };
    PrefixStore::Options o;
    o.dir=dir; o.identity="engine A"; o.disk_budget=size_t(1)<<30; o.bounce=4096;   // many chunks per payload
    ConversationCheckpoint root;
    SavedConversation ref;
    at_root(13,40,root,ref);
    {
        PrefixStore store;
        check(store.open(o,log,err) && store.size()==0,"open empty directory");
        check(store.capture(root,true,ss,g,draft.state,log,err) && store.size()==1,"capture straight to disk");
        check(store.contains(root.ids,true) && !store.contains(root.ids,false),"cvec is part of the identity");
        check(store.best(ids,true).tokens==40 && store.best(ids,false).tokens==0,"longest prefix match");
        check(store.best(root.ids,true).tokens==0,"never the whole prompt");
        fill(177);
        check(store.restore(store.best(ids,true).key,ss,g,draft.state,source,err)==ConversationRestore::restored &&
              source=="disk","no RAM budget: streamed from the file");
        check(restored(ref),"streamed restore is exact");
    }
    o.ram_budget=size_t(1)<<30;
    {
        PrefixStore store;
        check(store.open(o,log,err) && store.size()==1,"the index survives a restart");
        const auto key=store.best(ids,true).key;
        fill(91);
        check(store.restore(key,ss,g,draft.state,source,err)==ConversationRestore::restored && source=="disk>ram",
              "read into the RAM tier");
        check(restored(ref) && store.ram_bytes()>0,"exact through the RAM tier");
        fill(92);
        check(store.restore(key,ss,g,draft.state,source,err)==ConversationRestore::restored && source=="ram",
              "then from RAM");
        check(restored(ref),"exact from RAM");
        // a disk budget for one: the second prefix pushes out the least recently used
        ConversationCheckpoint root2;
        SavedConversation ref2;
        at_root(29,20,root2,ref2);
        PrefixStore::Options one=o;
        one.disk_budget=store.disk_bytes()+store.disk_bytes()/4;
        PrefixStore small;
        check(small.open(one,log,err) && small.size()==1,"reopen with a small budget");
        check(small.capture(root2,true,ss,g,draft.state,log,err) && small.size()==1 &&
              small.best(ids,true).tokens==20,"the older prefix leaves for the newer");
        fill(93);
        check(small.restore(small.best(ids,true).key,ss,g,draft.state,source,err)==ConversationRestore::restored &&
              restored(ref2),"the newer prefix restores");
    }
    {
        PrefixStore::Options other=o;
        other.identity="engine B";
        PrefixStore store;
        check(store.open(other,log,err) && store.size()==0,"another engine or model deletes the files");
        check(std::distance(fs::directory_iterator(dir),fs::directory_iterator{})==0,"and nothing is left");
        check(store.capture(root,true,ss,g,draft.state,log,err),"capture again");
    }
    {
        PrefixStore::Options other=o;
        other.identity="engine B";
        for (const auto& d : fs::directory_iterator(dir)) fs::resize_file(d.path(),fs::file_size(d.path())-1);
        PrefixStore store;
        check(store.open(other,log,err) && store.size()==0,"a file cut short is dropped");
    }
    fs::remove_all(dir);
}
}

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    const std::string dir=(std::filesystem::temp_directory_path()/"strata_prefix_store_test").string();
    for (int fmt : {kKvF16,kKvInt8,kKvQ4}) for (int mode : {0,1}) prefix_session(fmt,mode,dir);
    prefix_session(3,0,dir);
    std::printf("prefix_store_test: %d checks passed\n",checks);
}
