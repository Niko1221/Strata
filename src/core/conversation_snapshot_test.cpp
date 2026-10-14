#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/kernels/kv_q4.hpp"
#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <cstdlib>
#include <limits>
#include <new>
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
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

void full_session(int fmt, int mode, int experts) {
    Fixture main(fmt,mode), draft(fmt==3?kKvInt8:fmt,2);
    auto& g=main.g;
    g.n_layers=4; g.n_expert=experts;
    g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    SessionState ss;
    ss.max_cells=96; ss.qsa_states=&main.state;
    ss.layer_hi=g.n_layers; ss.gdn_alloc=g.n_gdn_layers(); ss.qsa_alloc=g.n_qsa_layers();   // the whole-model carve
    std::string err;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g,sizes,err),"whole-session geometry sizes");
    main.alloc(ss.gdn_state,sizes.gdn); main.alloc(ss.ple_hist,sizes.ple);
    main.alloc(main.state.idx_tail,sizes.tail); main.alloc(main.state.idx_dead,sizes.dead);
    main.alloc(main.state.idx_block_pos,sizes.block_pos);
    std::vector<int32_t> ids(65);
    for (size_t i=0;i<ids.size();++i) ids[i]=(int32_t)i+1;
    std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;
    auto fill=[&](uint8_t salt) {
        main.fill(salt); draft.fill(salt);
        for (const auto& [p,n] : std::vector<std::pair<void*,size_t>>{
                 {ss.gdn_state,sizes.gdn},{ss.ple_hist,sizes.ple},{main.state.idx_tail,sizes.tail},
                 {main.state.idx_dead,sizes.dead},{main.state.idx_block_pos,sizes.block_pos}})
            cuda_check(cudaMemset(p,salt,n));
        // A real indexer maintains its spare pooled row as a copy of idx_dead.
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        cuda_check(cudaDeviceSynchronize());
    };
    fill(13);
    ConversationCheckpoint checkpoint;
    checkpoint.ids.assign(ids.begin(),ids.begin()+3);
    check(conversation_checkpoint_save(checkpoint,ss,g,err),"save complete running checkpoint");
    checkpoint.used = 17;
    checkpoints.push_back(std::move(checkpoint));
    const ConversationView view{ids,images,checkpoints,true};
    SavedConversation a,b,restored;
    check(conversation_snapshot_save(a,view,ss,g,draft.state,err),"capture complete A");
    check(a.checkpoints[0].used == 17,"upstream checkpoint LRU stamp survives capture");
    fill(177);
    check(conversation_snapshot_save(b,view,ss,g,draft.state,err),"capture complete B");
    {
        // disk save path: metadata + streamed K/V give the same file as the captured image
        SavedConversation meta;
        std::vector<SessionKvSource> sources;
        check(conversation_snapshot_sources(meta,sources,view,ss,g,draft.state,err),"disk-save sources");
        check(meta.kv.empty() && sources.size()==b.kv.size(),"one source per K/V layer, none captured");
        const SessionFileIdentity id{1,2};
        namespace fs=std::filesystem;
        const fs::path dir=fs::temp_directory_path()/("strata-snap-test-"+std::to_string(experts)+"-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
        const std::string p1=(dir/"streamed.bin").string(), p2=(dir/"captured.bin").string();
        auto slurp=[](const std::string& p){ std::ifstream f(p,std::ios::binary);
            return std::vector<char>((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>()); };
        size_t n1=0,n2=0;
        check(session_file_write(p1,meta,sources,id,n1,err),"streamed session write");
        check(session_file_write(p2,b,id,n2,err),"captured session write");
        SavedConversation x,y;
        check(session_file_read(p1,id,x,n1,err) && session_file_read(p2,id,y,n2,err) && n1==n2,"both files read");
        check(x.live.gdn==y.live.gdn && x.live.dead==y.live.dead && x.live.ids==y.live.ids &&
              x.checkpoints.size()==y.checkpoints.size() && equal(x.kv[0],y.kv[0]) && equal(x.kv.back(),y.kv.back()),
              "streamed file equals captured file");
        check(slurp(p1)==slurp(p2),"streamed file is byte-identical to the captured file");
        std::error_code ec;
        fs::remove_all(dir,ec);
        auto empty_ids=std::vector<int32_t>{};
        const ConversationView none{empty_ids,images,checkpoints,true};
        check(!conversation_snapshot_sources(meta,sources,none,ss,g,draft.state,err),"empty session refused");
    }
    check(a.live.dead!=b.live.dead,"different opening-state spare keys in regression fixture");
    auto bad=a; bad.kv.back().k.pop_back();
    check(conversation_snapshot_restore(bad,ss,g,draft.state,err)==ConversationRestore::invalid,
          "reject invalid late draft before any main-layer write");
    check(conversation_snapshot_save(restored,view,ss,g,draft.state,err),"capture B after refused restore");
    check(restored.live.gdn==b.live.gdn && restored.live.dead==b.live.dead &&
          equal(restored.kv[0],b.kv[0]) && equal(restored.kv[1],b.kv[1]),"refusal preserves all tested state");
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore complete A");
    check(conversation_snapshot_save(restored,view,ss,g,draft.state,err),"capture restored A");
    check(restored.live.gdn==a.live.gdn && restored.live.ple==a.live.ple && restored.live.tails==a.live.tails &&
          restored.live.dead==a.live.dead && restored.live.block_pos==a.live.block_pos &&
          equal(restored.kv[0],a.kv[0]) && equal(restored.kv[1],a.kv[1]),"whole-session A/B/A exactness including spare key");
    check(ss.ple_prev[0]==64 && ss.ple_prev[1]==65,"PLE token window reconstructed");
    {
        // ---- the running-state codec (strata/core/state_codec.hpp). The state the session holds is fp32; a stored
        // copy may hold it as bf16. The session itself is never coded, and a coded copy is only ever read by a
        // restore, which widens it. Everything below compares against image `a`, the same state captured as fp32.
        const ConversationView coded_view{ids,images,checkpoints,true,StateCodec::Bf16};
        size_t plain=0,coded=0;
        check(conversation_snapshot_bytes(view,ss,g,draft.state,plain,err) &&
              conversation_snapshot_bytes(coded_view,ss,g,draft.state,coded,err),"a coded capture is estimable");
        SavedConversation c;
        check(conversation_snapshot_save(c,coded_view,ss,g,draft.state,err),"capture with the state coded");
        // the estimate is what the caller admits before it captures: it must cover what the capture allocates,
        // including the fp32 slice the coded copy crosses the device boundary through
        check(c.bytes()<=coded,"the coded capture stays inside its admitted bound");
        check(coded<=plain+std::min(sizes.gdn,(size_t) kStateCodecChunkFloats*sizeof(float)),
              "and a coded capture costs no more than the fp32 one plus one slice");
        // ...and the estimate follows the codec into the LIVE state and not only into the checkpoints. An estimate
        // that still charged the live arrays at fp32 asked the cache for about twice what the park needs and refused
        // parks that fit (a 116.6 MiB park turned away from a 128 MiB cache for being 172.9 MiB). The drop from the
        // fp32 estimate is every coded array's own half, less the one slice held while it crosses the device.
        const size_t slice = std::min(sizes.gdn, (size_t) kStateCodecChunkFloats * sizeof(float));
        const size_t live_fp32 = sizes.gdn + (ss.ple_hist ? sizes.ple : 0)
                               + (size_t) g.n_qsa_layers() * (sizes.tail + sizes.dead);
        size_t ckpt_fp32 = 0;
        for (const auto& k : checkpoints)
            ckpt_fp32 += k.gdn.size() + k.ple.size() + k.tails.size() + k.dead.size();
        check(plain > coded && plain - coded + slice >= (live_fp32 + ckpt_fp32) / 2,
              "the estimate prices the live running state the way the capture stores it");
        check(c.live.state_codec==(uint8_t) StateCodec::Bf16 && c.checkpoints[0].state_codec==(uint8_t) StateCodec::Bf16,
              "the image says, on every checkpoint, how its running state is held");
        check(c.live.gdn.size()*2==a.live.gdn.size() && c.live.ple.size()*2==a.live.ple.size() &&
              c.live.tails.size()*2==a.live.tails.size() && c.live.dead.size()*2==a.live.dead.size(),
              "the coded arrays are half the fp32 ones");
        check(c.live.block_pos==a.live.block_pos,"block_pos is int32 positions and is never coded");
        check(c.bytes()<a.bytes(),"and the image is smaller than the same conversation held as fp32");
        check(conversation_snapshot_validate(c,ss,g,draft.state,err),"a coded image validates");
        auto relabelled=c; relabelled.live.state_codec=0;
        check(!conversation_snapshot_validate(relabelled,ss,g,draft.state,err),
              "a coded image read as fp32 is refused: its arrays are the wrong size for fp32");
        // widen the fp32 image's own arrays and compare value by value: that is what the restore puts in the session
        auto widened=[](const std::vector<uint8_t>& coded){
            std::vector<float> f(coded.size()/2);
            std::string e;
            check(state_decode(coded.data(),f.size(),StateCodec::Bf16,f.data(),e),"a coded array decodes");
            return f;
        };
        auto within=[](const std::vector<uint8_t>& fp32,const std::vector<float>& got){
            const float* want=(const float*) fp32.data();
            for(size_t i=0;i<got.size();++i){
                const double a=want[i],b=got[i];
                if(a==0.0f){ if(b!=0.0f) return false; continue; }
                if(std::fabs((b-a)/a)>1.0/256) return false;
            }
            return true;
        };
        check(within(a.live.gdn,widened(c.live.gdn)) && within(a.live.dead,widened(c.live.dead)) &&
              within(a.live.tails,widened(c.live.tails)) && within(a.live.ple,widened(c.live.ple)),
              "every coded value is within 2^-8 of the fp32 value it stands for");
        // restore it: the session's fp32 buffers get the widened values, and re-capturing as fp32 finds exactly them
        check(conversation_snapshot_restore(c,ss,g,draft.state,err)==ConversationRestore::restored,
              "restore an image whose running state is coded");
        SavedConversation back;
        check(conversation_snapshot_save(back,view,ss,g,draft.state,err),"capture again as fp32");
        check(back.live.gdn.size()==c.live.gdn.size()*2 &&
              std::equal(back.live.gdn.begin(),back.live.gdn.end(),a.live.gdn.begin())==false,
              "the restored session holds the widened state, not the fp32 bytes it came from");
        check(back.live.block_pos==a.live.block_pos && back.live.ids==a.live.ids &&
              back.kv[0].k==a.kv[0].k && back.kv[0].v==a.kv[0].v &&
              back.kv[0].k_scale==a.kv[0].k_scale && back.kv[0].v_scale==a.kv[0].v_scale,
              "everything that is not running state came back exactly: the K/V pages and their scales");
        // One thing does follow the coding, and it should: the indexer's spare pooled row is put back from the
        // checkpoint's dead keys, which are running state. It is the only K/V plane that can differ.
        check(equal(back.kv[0],a.kv[0]) || back.kv[0].pooled!=a.kv[0].pooled,
              "if a K/V plane differs after a coded restore it is the pooled one, the row that comes from state");
        // and coding that state again is stable: bf16 of a bf16 value is that value
        SavedConversation again;
        check(conversation_snapshot_save(again,coded_view,ss,g,draft.state,err),"capture the restored state coded again");
        check(again.live.gdn==c.live.gdn && again.live.dead==c.live.dead && again.live.tails==c.live.tails &&
              again.live.ple==c.live.ple,"coding a coded value changes nothing: a park does not degrade a park");
        // the checkpoints on their own: store keeps a coded chain as it is, widen puts it back to fp32
        std::vector<ConversationCheckpoint> kept,wide;
        check(conversation_checkpoints_store(c.checkpoints,StateCodec::Bf16,kept,err) &&
              kept[0].gdn==c.checkpoints[0].gdn && kept[0].state_codec==c.checkpoints[0].state_codec,
              "a chain already under the codec is copied, not coded twice");
        check(conversation_checkpoints_store(a.checkpoints,StateCodec::Bf16,kept,err) &&
              kept[0].gdn.size()*2==a.checkpoints[0].gdn.size() && kept[0].state_codec==(uint8_t) StateCodec::Bf16,
              "an fp32 chain is coded by a stored copy");
        check(conversation_checkpoints_store(a.checkpoints,StateCodec::Fp32,kept,err) &&
              kept[0].gdn==a.checkpoints[0].gdn && kept[0].state_codec==0,
              "and with no codec asked for it is the plain copy it has always been");
        wide=c.checkpoints;
        auto floats_of=[](const std::vector<uint8_t>& fp32){
            std::vector<float> f(fp32.size()/4);
            std::memcpy(f.data(),fp32.data(),f.size()*sizeof(float));
            return f;
        };
        check(conversation_checkpoints_widen(wide,err) && wide[0].state_codec==0 &&
              wide[0].gdn.size()==c.checkpoints[0].gdn.size()*2 && wide[0].block_pos==a.checkpoints[0].block_pos &&
              wide[0].ids==a.checkpoints[0].ids && within(a.checkpoints[0].gdn,floats_of(wide[0].gdn)) &&
              within(a.checkpoints[0].dead,floats_of(wide[0].dead)),
              "a mount widens a parked chain back to fp32: the sizes and the bytes that are not coded return "
              "exactly, every coded value within 2^-8 of what the chain held");
        wide=a.checkpoints;
        check(conversation_checkpoints_widen(wide,err) && wide[0].gdn==a.checkpoints[0].gdn,
              "widening a chain that is already fp32 leaves every byte alone");
        // a mount widens in place: the array it reads is the array it writes. The bytes must be the ones the same
        // chain gives when it is held in a separate buffer, and they must be the same on a second pass.
        {
            std::vector<ConversationCheckpoint> inplace=c.checkpoints, stored;
            check(conversation_checkpoints_store(c.checkpoints,StateCodec::Bf16,stored,err) &&
                  conversation_checkpoints_widen(stored,err),
                  "the same chain widened from a buffer of its own");
            check(conversation_checkpoints_widen(inplace,err),
                  "and the same chain widened in place, source and destination one vector");
            check(stored[0].gdn==inplace[0].gdn && stored[0].ple==inplace[0].ple &&
                  stored[0].tails==inplace[0].tails && stored[0].dead==inplace[0].dead &&
                  stored[0].block_pos==inplace[0].block_pos && stored[0].state_codec==inplace[0].state_codec,
              "both give the same fp32 bytes, over every slice of an array far larger than the chunk size");
        }
        // a coded session file: version 2, and the streamed and captured paths still agree
        {
            SavedConversation meta; std::vector<SessionKvSource> sources;
            check(conversation_snapshot_sources(meta,sources,coded_view,ss,g,draft.state,err),"disk-save sources, coded");
            check(meta.live.state_codec==(uint8_t) StateCodec::Bf16,"the disk-save path codes the live state too");
            namespace fs=std::filesystem;
            const fs::path dir=fs::temp_directory_path()/("strata-snap-coded-"+std::to_string(experts)+"-"+
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            fs::create_directories(dir);
            const SessionFileIdentity id{1,2};
            const std::string p1=(dir/"streamed.bin").string(),p2=(dir/"captured.bin").string(),
                            p3=(dir/"plain.bin").string();
            auto slurp=[](const std::string& p){ std::ifstream f(p,std::ios::binary);
                return std::vector<char>((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>()); };
            size_t n1=0,n2=0,n3=0;
            check(session_file_write(p1,meta,sources,id,n1,err) && session_file_write(p2,again,id,n2,err),
                  "both coded writers succeed");
            check(slurp(p1)==slurp(p2),"a coded streamed file is byte-identical to the coded captured file");
            check(uint8_t(slurp(p1)[8])==2,"a coded session file is version 2");
            check(session_file_write(p3,a,id,n3,err) && uint8_t(slurp(p3)[8])==1,
                  "the same conversation with fp32 state is still a version 1 file");
            check(n1<n3,"and it is smaller than that file");
            SavedConversation x;
            check(session_file_read(p1,id,x,n1,err) && x.live.gdn==again.live.gdn &&
                  x.live.state_codec==(uint8_t) StateCodec::Bf16 && x.checkpoints[0].state_codec==x.live.state_codec,
                  "a coded session file reads back with its codec and its bytes");
            check(conversation_snapshot_restore(x,ss,g,draft.state,err)==ConversationRestore::restored,
                  "and restores");
            std::error_code ec; fs::remove_all(dir,ec);
        }
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,
              "the session is back to the fp32 image for the checks that follow");
    }
    for (int64_t dirty : {65, 3, 0}) {
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore growth fixture base");
        ConversationKvReuse reuse{a.kv,65,dirty};
        const uint8_t* original = nullptr;
        reuse.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){original=p;return true;});
        main.fill_after(91,dirty); draft.fill_after(91,std::max<int64_t>(0,dirty-1));
        cuda_check(cudaMemset(ss.gdn_state,91,sizes.gdn));
        ids.resize(70);
        for (size_t i=65;i<ids.size();++i) ids[i]=int32_t(i+1);
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        SavedConversation fresh,incremental;
        check(conversation_snapshot_save(fresh,view,ss,g,draft.state,err),"full capture reference after growth or rewind");
        size_t peak=0,reused=0;
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.state,peak,err),"admit incremental capture peak");
        check(conversation_snapshot_save(incremental,view,ss,g,draft.state,err,std::move(reuse),&reused),"capture with retained pages");
        check(incremental.bytes() <= peak,"incremental allocation stays within admitted bound");
        check(incremental.live.gdn==fresh.live.gdn && incremental.live.ple==fresh.live.ple &&
              incremental.live.dead==fresh.live.dead && equal(incremental.kv[0],fresh.kv[0]) &&
              equal(incremental.kv[1],fresh.kv[1]),"incremental capture equals full capture after growth or rewind");
        check((dirty>=4)==(reused>0),"only complete unchanged pages or rows are retained");
        incremental.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){
            check(p==original,"growth never reallocates the retained payload");return true;
        });
        check(conversation_snapshot_restore(incremental,ss,g,draft.state,err)==ConversationRestore::restored,"restore segmented incremental snapshot");
        uint64_t fingerprint=0;
        check(conversation_kv_verify(incremental.kv.back(),draft.state,g,70,false,fingerprint,err),"incremental draft authoritative and ring read-back");
        ids.resize(65);
    }
    {
        // a layer split's later stage: the same image WITHOUT the draft layer's K/V (draft == nullptr)
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A before stage images");
        size_t with_draft=0,without=0;
        check(conversation_snapshot_bytes(view,ss,g,draft.state,with_draft,err) &&
              conversation_snapshot_bytes(view,ss,g,nullptr,without,err) && without<with_draft,
              "a stage image's estimate leaves the draft out");
        SavedConversation stage,stage_b,stage_back;
        check(conversation_snapshot_save(stage,view,ss,g,nullptr,err),"capture a stage image");
        check(stage.kv.size()==a.kv.size()-1 && equal(stage.kv[0],a.kv[0]) && stage.live.gdn==a.live.gdn,
              "a stage image holds the session's own layers, no draft");
        check(stage.bytes()<a.bytes(),"a stage image is smaller by the draft ring");
        check(conversation_snapshot_validate(stage,ss,g,nullptr,err),"validate a stage image without a draft");
        check(!conversation_snapshot_validate(stage,ss,g,draft.state,err),"a stage image is refused where a draft is expected");
        check(!conversation_snapshot_validate(a,ss,g,nullptr,err),"an image with a draft is refused where none is expected");
        fill(201);
        check(conversation_snapshot_save(stage_b,view,ss,g,nullptr,err),"capture another stage state");
        check(conversation_snapshot_restore(a,ss,g,nullptr,err)==ConversationRestore::invalid,
              "restoring a draft image as a stage image is refused before any write");
        check(conversation_snapshot_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage_b.live.gdn &&
              equal(stage_back.kv[0],stage_b.kv[0]),"the refusal left the stage untouched");
        check(conversation_snapshot_restore(stage,ss,g,nullptr,err)==ConversationRestore::restored,"restore a stage image");
        check(conversation_snapshot_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage.live.gdn &&
              stage_back.live.dead==stage.live.dead && equal(stage_back.kv[0],stage.kv[0]),"stage A/B/A exactness");
        // its retained K/V: the next park copies only what changed, and equals a full capture
        ConversationKvReuse reuse{stage.kv,65,65,{}};
        SavedConversation fresh,incremental;
        size_t peak=0,reused=0;
        check(conversation_snapshot_save(fresh,view,ss,g,nullptr,err),"full stage capture reference");
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,nullptr,peak,err),"admit an incremental stage capture");
        check(conversation_snapshot_save(incremental,view,ss,g,nullptr,err,std::move(reuse),&reused),"incremental stage capture");
        check(reused>0 && incremental.bytes()<=peak && equal(incremental.kv[0],fresh.kv[0]) &&
              incremental.live.gdn==fresh.live.gdn,"an incremental stage capture reuses pages and equals a full one");
        SavedConversation split_image=stage;
        split_image.stage_images.push_back(stage);
        check(!conversation_snapshot_validate(split_image,ss,g,draft.state,err),
              "a layer split's image is refused by the whole-session form");
        ConversationKvReuse wrong{a.kv,65,65,{}};
        check(!conversation_snapshot_capture_bytes(wrong,view,ss,g,nullptr,peak,err),"a draft image's K/V is not a stage's reuse");
    }
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A after stage images");
    check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"restore early running checkpoint");
    std::vector<uint8_t> spare(sizes.dead);
    cuda_check(cudaMemcpy(spare.data(),main.state.idx_pooled,sizes.dead,cudaMemcpyDeviceToHost));
    check(spare==a.checkpoints[0].dead,"checkpoint rebuilds spare row over a later completed block");
    check(ss.ple_prev[0]==2 && ss.ple_prev[1]==3,"checkpoint PLE token window");
    // ---- lazy restoration of the retained chain. Resuming a parked conversation no longer widens the whole chain:
    // its snapshots join the active chain in the codec they were parked in, an append-only continuation leaves them
    // there, and the ONE snapshot a rewind selects is decoded into the session's fp32 buffers while its stored bytes
    // stay exactly as they were. Every value check below is against `plain`, the same points captured as fp32.
    {
        auto f32=[](const std::vector<uint8_t>& fp32){
            std::vector<float> f(fp32.size()/4);
            std::memcpy(f.data(),fp32.data(),f.size()*sizeof(float));
            return f;
        };
        auto close=[](const std::vector<uint8_t>& want_fp32,const std::vector<float>& got){
            const float* want=(const float*) want_fp32.data();
            if(got.size()!=want_fp32.size()/4) return false;
            for(size_t i=0;i<got.size();++i){
                const double x=want[i],y=got[i];
                if(x==0.0f){ if(y!=0.0f) return false; continue; }
                if(std::fabs((y-x)/x)>1.0/256) return false;
            }
            return true;
        };
        std::vector<int32_t> tok(65);
        for(size_t i=0;i<tok.size();++i) tok[i]=(int32_t)i+1;
        std::vector<ConversationImageKey> none;
        const std::vector<int64_t> at{16,32,48};
        std::vector<ConversationCheckpoint> plain,coded;
        for(size_t i=0;i<at.size();++i){
            fill((uint8_t)(21+i*61));   // a different running state behind every checkpoint
            ConversationCheckpoint p;
            p.ids.assign(tok.begin(),tok.begin()+at[i]);
            check(conversation_checkpoint_save(p,ss,g,err),"lazy: capture an fp32 checkpoint");
            plain.push_back(std::move(p));
            ConversationCheckpoint q;
            q.ids=plain.back().ids;
            q.state_codec=(uint8_t) StateCodec::Bf16;
            check(conversation_checkpoint_save(q,ss,g,err),"lazy: capture the same point coded");
            check(q.state_codec==(uint8_t) StateCodec::Bf16 && q.gdn.size()*2==plain.back().gdn.size() &&
                  q.ple.size()*2==plain.back().ple.size() && q.tails.size()*2==plain.back().tails.size() &&
                  q.dead.size()*2==plain.back().dead.size() && q.block_pos==plain.back().block_pos,
                  "lazy: the coded checkpoint holds half the fp32 one, and block_pos is not coded");
            coded.push_back(std::move(q));
        }
        // the parked image a conversation cache would hand back, and the mount that hands its chain to the engine
        const ConversationView parked_view{tok,none,coded,true,StateCodec::Bf16};
        SavedConversation parked;
        check(conversation_snapshot_save(parked,parked_view,ss,g,draft.state,err),"lazy: park a coded chain");
        check(parked.checkpoints.size()==coded.size(),"lazy: the parked image holds the whole retained chain");
        std::vector<std::vector<uint8_t>> kept;
        for(const auto& c:parked.checkpoints) kept.push_back(c.gdn);
        std::vector<ConversationCheckpoint> chain=std::move(parked.checkpoints);
        for(size_t i=0;i<chain.size();++i)
            check(chain[i].state_codec==(uint8_t) StateCodec::Bf16 && chain[i].gdn==kept[i] &&
                  chain[i].gdn.size()*2==plain[i].gdn.size() && chain[i].block_pos==plain[i].block_pos &&
                  chain[i].ids==plain[i].ids,
                  "lazy: a resumed chain keeps every imported snapshot's bytes and its codec, unwidened");
        // an append-only continuation: the conversation keeps generating and captures its own checkpoints. Nothing
        // here touches the older, coded ones.
        fill(91);
        ConversationCheckpoint fresh;
        fresh.ids=tok;
        check(conversation_checkpoint_save(fresh,ss,g,err),"lazy: the continuing conversation captures a checkpoint");
        // the core API keeps the bytes the caller asked for: left at 0 the capture is fp32. The serve loop asks
        // for its own coding by setting `state_codec` first, which is what fresh_capture() below checks.
        check(fresh.state_codec==0,"lazy: a capture left at the API default is fp32, and only then");
        const auto before=chain;
        chain.push_back(std::move(fresh));
        bool older_untouched=chain.size()==before.size()+1;
        for(size_t i=0;i<before.size();++i)
            older_untouched=older_untouched && chain[i].state_codec==(uint8_t) StateCodec::Bf16 &&
                            chain[i].gdn==before[i].gdn && chain[i].ple==before[i].ple &&
                            chain[i].tails==before[i].tails && chain[i].dead==before[i].dead &&
                            chain[i].bytes()==before[i].bytes();
        check(older_untouched,"lazy: continuing on does not widen the older retained checkpoints");
        size_t mixed_bytes=0;
        const ConversationView mixed_view{tok,none,chain,true};
        check(conversation_snapshot_bytes(mixed_view,ss,g,draft.state,mixed_bytes,err) && mixed_bytes>0,
              "lazy: a chain holding fp32 and coded snapshots together is estimable");
        // select an older coded checkpoint: the session's fp32 buffers get the widened values, the snapshot does not
        fill(200);
        check(conversation_checkpoint_restore(chain[0],ss,g,err),"lazy: restore a retained coded checkpoint");
        ConversationCheckpoint widened;
        widened.ids=plain[0].ids;
        check(conversation_checkpoint_save(widened,ss,g,err),"lazy: capture what the restore put in the session");
        check(widened.state_codec==0 && close(plain[0].gdn,f32(widened.gdn)) &&
              close(plain[0].dead,f32(widened.dead)) && close(plain[0].tails,f32(widened.tails)) &&
              close(plain[0].ple,f32(widened.ple)) && widened.block_pos==plain[0].block_pos,
              "lazy: the selected snapshot restores the correctly widened fp32 values");
        check(chain[0].state_codec==(uint8_t) StateCodec::Bf16 && chain[0].gdn==kept[0] &&
              chain[0].bytes()==before[0].bytes(),
              "lazy: restoring a checkpoint leaves the retained snapshot coded, not permanently widened");
        // the mixed chain, re-parked and saved both ways, and restored again
        {
            const ConversationView fp32_view{tok,none,chain,true};
            const ConversationView bf16_view{tok,none,chain,true,StateCodec::Bf16};
            SavedConversation as_fp32,as_bf16;
            check(conversation_snapshot_save(as_fp32,fp32_view,ss,g,draft.state,err),
                  "lazy: park a mixed chain with the engine configured fp32");
            check(conversation_snapshot_save(as_bf16,bf16_view,ss,g,draft.state,err),
                  "lazy: and with it configured bf16");
            bool all_fp32=true,all_bf16=true;
            for(const auto& c:as_fp32.checkpoints)
                all_fp32=all_fp32 && c.state_codec==0 && c.gdn.size()==plain[0].gdn.size();
            for(size_t i=0;i<as_bf16.checkpoints.size();++i)
                all_bf16=all_bf16 && as_bf16.checkpoints[i].state_codec==(uint8_t) StateCodec::Bf16 &&
                         as_bf16.checkpoints[i].gdn.size()==plain[0].gdn.size()/2;
            check(all_fp32,"lazy: an fp32 write expands the coded snapshots it is given");
            check(all_bf16 && as_bf16.checkpoints[0].gdn==kept[0],
                  "lazy: a bf16 write copies a snapshot that is already bf16, byte for byte");
            // a chain that really is mixed once stored: one coded snapshot among fp32 ones
            SavedConversation mix=as_fp32;
            mix.checkpoints[0]=as_bf16.checkpoints[0];
            check(conversation_snapshot_validate(mix,ss,g,draft.state,err),
                  "lazy: an image holding both fp32 and coded snapshots validates, checkpoint by checkpoint");
            check(conversation_snapshot_restore(mix,ss,g,draft.state,err)==ConversationRestore::restored,
                  "lazy: and restores");
            ConversationCheckpoint again; again.ids=plain[0].ids;
            check(conversation_checkpoint_save(again,ss,g,err),"lazy: capture the state a mixed image restored");
            check(close(plain[0].gdn,f32(again.gdn)) && again.block_pos==plain[0].block_pos,
                  "lazy: its coded snapshot came back widened and its fp32 ones came back exactly");
            // SAVE: the file's own representation is the configured one, whichever checkpoint it selects
            chain[0].pinned=true;   // so the SAVE selects a coded checkpoint out of the mixed chain
            SessionSaveLive sl; sl.state_bytes=sizes.gdn+sizes.ple; sl.tokens=(uint64_t) tok.size();
            sl.images=0; sl.kv_layers=1;
            std::vector<ConversationCheckpoint> disk; std::string why;
            const auto agree=[&](uint64_t,std::string&){ return true; };
            check(session_save_checkpoints(chain,sl,StateCodec::Fp32,agree,disk,why) && disk.size()==1 &&
                  disk[0].state_codec==(uint8_t) StateCodec::Bf16,
                  "lazy: the SAVE's checkpoint list keeps the selected snapshot as the chain holds it");
            check(session_save_peak_bytes(&disk[0],sl,StateCodec::Fp32)>2*(uint64_t) disk[0].bytes(),
                  "lazy: and its preflight prices the widening an fp32 file still has to do");
            const ConversationView disk_view{tok,none,disk,true,StateCodec::Fp32};
            SavedConversation dmeta; std::vector<SessionKvSource> dsrc;
            check(conversation_snapshot_sources(dmeta,dsrc,disk_view,ss,g,draft.state,err) &&
                  dmeta.checkpoints[0].state_codec==0 && dmeta.checkpoints[0].gdn.size()==plain[0].gdn.size(),
                  "lazy: an fp32 SAVE of a coded retained snapshot writes fp32 bytes");
            chain[0].pinned=false;
            // and on disk: the mixed image says, per checkpoint, how each one is held
            namespace fs=std::filesystem;
            const fs::path dir=fs::temp_directory_path()/("strata-lazy-"+std::to_string(experts)+"-"+
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            fs::create_directories(dir);
            const SessionFileIdentity id{1,2};
            auto version_of=[](const std::string& p){ std::ifstream f(p,std::ios::binary);
                std::vector<char> b(12); f.read(b.data(),12); return (uint8_t) b[8]; };
            size_t n1=0,n2=0,n3=0;
            const std::string p1=(dir/"all-fp32.bin").string(),p2=(dir/"all-bf16.bin").string(),
                            p3=(dir/"mixed.bin").string();
            SavedConversation m3; std::vector<SessionKvSource> s3;
            check(conversation_snapshot_sources(m3,s3,fp32_view,ss,g,draft.state,err) &&
                  session_file_write(p1,m3,s3,id,n1,err),"lazy: a SAVE of the mixed chain configured fp32 writes");
            check(version_of(p1)==1,"lazy: an all-fp32 SAVE of a mixed chain is still a version 1 file");
            check(session_file_write(p2,as_bf16,id,n2,err) && version_of(p2)==2,
                  "lazy: and configured bf16 it is a version 2 file");
            check(session_file_write(p3,mix,id,n3,err) && version_of(p3)==2,
                  "lazy: a mixed image on disk is version 2 and says so per checkpoint");
            SavedConversation r3;
            size_t n4=0;
            check(session_file_read(p3,id,r3,n4,err) && r3.checkpoints[0].state_codec==(uint8_t) StateCodec::Bf16 &&
                  r3.checkpoints[0].gdn==kept[0] && r3.checkpoints[1].state_codec==0 &&
                  r3.checkpoints[1].gdn==as_fp32.checkpoints[1].gdn,
                  "lazy: and it reads back with each checkpoint's own coding");
            check(conversation_snapshot_restore(r3,ss,g,draft.state,err)==ConversationRestore::restored,
                  "lazy: a chain read back mixed restores");
            std::error_code ec; fs::remove_all(dir,ec);
        }
        // the deferred restore's refusals: a snapshot that lies about its coding is refused before anything is written
        fill(57);
        ConversationCheckpoint probe; probe.ids=plain[1].ids;
        check(conversation_checkpoint_save(probe,ss,g,err),"lazy: reference state before the refused restores");
        auto untouched=[&](){
            ConversationCheckpoint now; now.ids=plain[1].ids;
            return conversation_checkpoint_save(now,ss,g,err) && now.gdn==probe.gdn && now.dead==probe.dead &&
                   now.tails==probe.tails && now.block_pos==probe.block_pos;
        };
        auto bad=chain[1];
        bad.state_codec=7;   // a codec this build does not know
        err.clear();
        check(!conversation_checkpoint_restore(bad,ss,g,err) && !err.empty(),
              "lazy: a retained snapshot naming an unknown codec is refused, not read as fp32");
        check(untouched(),"lazy: and the refusal happens before any state write");
        bad=chain[1]; bad.gdn.pop_back();
        err.clear();
        check(!conversation_checkpoint_restore(bad,ss,g,err) && !err.empty(),
              "lazy: a coded snapshot whose arrays are not its codec's size is refused");
        check(untouched(),"lazy: the session still holds the state the last capture saw");
        // a layer split: a checkpoint's own coding and each stage part's travel through the split and the merge
        {
            fill(73);
            ConversationCheckpoint cp;
            cp.ids=plain[2].ids;
            cp.state_codec=(uint8_t) StateCodec::Bf16;   // stage 0's running state coded
            check(conversation_checkpoint_save(cp,ss,g,err),"lazy: capture a split checkpoint's first stage");
            std::vector<uint8_t> first=cp.gdn;
            for(size_t k=0;k<2;++k){
                ConversationCheckpoint part;
                part.ids=cp.ids;
                part.state_codec=(uint8_t)(k==1?StateCodec::Bf16:StateCodec::Fp32);
                check(conversation_checkpoint_save(part,ss,g,err),"lazy: capture one stage's part");
                cp.stage_parts.push_back(std::move(part));
            }
            std::vector<ConversationCheckpoint> in; in.push_back(std::move(cp));
            auto split=conversation_checkpoints_split(std::move(in),2);
            check(split.stage0.size()==1 && split.parts[0].size()==1 && split.parts[1].size()==1 &&
                  split.stage0[0].state_codec==(uint8_t) StateCodec::Bf16 && split.stage0[0].gdn==first &&
                  split.parts[0][0].state_codec==0 && split.parts[1][0].state_codec==(uint8_t) StateCodec::Bf16,
              "lazy: a split keeps a checkpoint's own coding and each stage part's separately");
            std::vector<ConversationCheckpoint> joined;
            check(conversation_checkpoints_merge(std::move(split),joined) && joined.size()==1 &&
                  joined[0].state_codec==(uint8_t) StateCodec::Bf16 && joined[0].gdn==first &&
                  joined[0].stage_parts.size()==2 && joined[0].stage_parts[0].state_codec==0 &&
                  joined[0].stage_parts[1].state_codec==(uint8_t) StateCodec::Bf16,
                  "lazy: and merging them back does too");
            check(conversation_checkpoint_restore(joined[0].stage_parts[1],ss,g,err),
                  "lazy: a coded stage part restores into its stage's fp32 buffers");
        }
    }
}
}

// ---- capture-time coding of RETAINED conversation checkpoints. `--session-state-codec bf16` decides how a stored
// copy of a conversation's running state is held, and a checkpoint the serve loop keeps in its chain IS a stored
// copy: the engine sets the checkpoint's own `state_codec` before capturing it, so the snapshot is coded the moment
// it is taken and not only when the conversation is later parked. Everything below is what that has to guarantee,
// checked before anything in this file parks or saves: the live buffers stay fp32 and untouched, `block_pos` is
// int32 positions and stays exact, a restore widens the ONE snapshot it selects and leaves the snapshot itself
// coded, and a capture that cannot allocate refuses with the checkpoints and the working state exactly as they were.
void fresh_capture(int experts) {
    Fixture main(kKvInt8,0), draft(kKvInt8,2);
    auto& g=main.g;
    g.n_layers=4; g.n_expert=experts;
    g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    SessionState ss;
    ss.max_cells=96; ss.qsa_states=&main.state;
    ss.layer_hi=g.n_layers; ss.gdn_alloc=g.n_gdn_layers(); ss.qsa_alloc=g.n_qsa_layers();
    std::string err;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g,sizes,err),"capture: geometry sizes");
    main.alloc(ss.gdn_state,sizes.gdn); main.alloc(ss.ple_hist,sizes.ple);
    main.alloc(main.state.idx_tail,sizes.tail); main.alloc(main.state.idx_dead,sizes.dead);
    main.alloc(main.state.idx_block_pos,sizes.block_pos);
    std::vector<int32_t> ids(65);
    for(size_t i=0;i<ids.size();++i) ids[i]=(int32_t)i+1;
    std::vector<ConversationImageKey> images;
    auto fill=[&](uint8_t salt) {
        main.fill(salt); draft.fill(salt);
        for (const auto& [p,n] : std::vector<std::pair<void*,size_t>>{
                 {ss.gdn_state,sizes.gdn},{ss.ple_hist,sizes.ple},{main.state.idx_tail,sizes.tail},
                 {main.state.idx_dead,sizes.dead},{main.state.idx_block_pos,sizes.block_pos}})
            cuda_check(cudaMemset(p,salt,n));
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        cuda_check(cudaDeviceSynchronize());
    };
    // the session's own fp32 buffers, read back: a capture reads them and never writes them
    auto live=[&](void* p, size_t n) {
        std::vector<uint8_t> b(n); cuda_check(cudaMemcpy(b.data(),p,n,cudaMemcpyDeviceToHost)); return b;
    };
    auto f32=[](const std::vector<uint8_t>& coded){
        std::vector<float> f(coded.size()/4);
        std::memcpy(f.data(),coded.data(),f.size()*sizeof(float));
        return f;
    };
    auto close=[](const std::vector<uint8_t>& want_fp32,const std::vector<float>& got){
        const float* want=(const float*) want_fp32.data();
        if(got.size()!=want_fp32.size()/4) return false;
        for(size_t i=0;i<got.size();++i){
            const double x=want[i],y=got[i];
            if(x==0.0f){ if(y!=0.0f) return false; continue; }
            if(std::fabs((y-x)/x)>1.0/256) return false;
        }
        return true;
    };
    // ---- 1. a freshly captured checkpoint under the option has the codec it was asked for and half the bytes,
    // and 2. the int32 positions in it are exact. No parking has happened anywhere in this function yet.
    fill(11);
    ConversationCheckpoint ref;
    ref.ids=ids;
    check(conversation_checkpoint_save(ref,ss,g,err),"capture: the same point as fp32 (the reference)");
    ConversationCheckpoint bf;
    bf.ids=ids;
    bf.state_codec=(uint8_t) StateCodec::Bf16;   // what the serve loop sets from --session-state-codec bf16
    const std::vector<uint8_t> gdn_before=live(ss.gdn_state,sizes.gdn);
    const std::vector<uint8_t> dead_before=live(main.state.idx_dead,sizes.dead);
    const std::vector<uint8_t> pos_before=live(main.state.idx_block_pos,sizes.block_pos);
    check(conversation_checkpoint_save(bf,ss,g,err),"capture: a retained checkpoint captured under bf16");
    check(bf.state_codec==(uint8_t) StateCodec::Bf16,
          "capture: a newly captured retained checkpoint names the codec it was captured in");
    check(bf.gdn.size()*2==ref.gdn.size() && bf.ple.size()*2==ref.ple.size() &&
          bf.tails.size()*2==ref.tails.size() && bf.dead.size()*2==ref.dead.size(),
          "capture: every coded array is the size that codec makes");
    check(bf.block_pos==ref.block_pos && bf.block_pos==pos_before && !bf.block_pos.empty(),
          "capture: block_pos is int32 positions and stays exact, byte for byte, under a coded capture");
    // ---- 3. the capture left the working state exactly as it found it
    check(live(ss.gdn_state,sizes.gdn)==gdn_before && live(main.state.idx_dead,sizes.dead)==dead_before &&
          live(main.state.idx_block_pos,sizes.block_pos)==pos_before,
          "capture: a coded capture writes the snapshot and not the session");
    // ---- 4. restoring it widens correctly and leaves the snapshot compressed, as many times as it is restored
    fill(199);   // the conversation moves on: a restore has to put the older values back
    const std::vector<uint8_t> kept_gdn=bf.gdn, kept_ple=bf.ple, kept_tails=bf.tails, kept_dead=bf.dead;
    check(conversation_checkpoint_restore(bf,ss,g,err),"capture: restore a freshly captured bf16 checkpoint");
    ConversationCheckpoint widened;
    widened.ids=ids;
    check(conversation_checkpoint_save(widened,ss,g,err),"capture: what that restore put in the session");
    check(widened.state_codec==0 && close(ref.gdn,f32(widened.gdn)) && close(ref.ple,f32(widened.ple)) &&
          close(ref.tails,f32(widened.tails)) && close(ref.dead,f32(widened.dead)) &&
          widened.block_pos==ref.block_pos,
          "capture: the freshly captured snapshot restores the correctly widened fp32 values");
    check(bf.state_codec==(uint8_t) StateCodec::Bf16 && bf.gdn==kept_gdn && bf.ple==kept_ple &&
          bf.tails==kept_tails && bf.dead==kept_dead,
          "capture: restoring it left the snapshot coded, not widened in place");
    fill(57);
    check(conversation_checkpoint_restore(bf,ss,g,err) && conversation_checkpoint_restore(bf,ss,g,err),
          "capture: the same coded snapshot restores again");
    check(bf.gdn==kept_gdn && bf.ple==kept_ple && bf.tails==kept_tails && bf.dead==kept_dead &&
          bf.block_pos==ref.block_pos,
          "capture: and repeated restoration does not change its stored bytes");
    // ---- 5. the chain this leaves: an imported coded snapshot beside one captured under the option beside one
    // captured with the option off. Branching, retention, a park and a SAVE all read each one as itself.
    fill(123);
    ConversationCheckpoint plain;
    plain.ids=ids;   // left at 0: what a capture with --session-state-codec fp32 (the default) produces
    check(conversation_checkpoint_save(plain,ss,g,err),"capture: a retained checkpoint with the option off");
    check(plain.state_codec==0 && plain.gdn.size()==ref.gdn.size(),
          "capture: with the option off a new checkpoint is the fp32 bytes it always was");
    std::vector<ConversationCheckpoint> chain;
    chain.push_back(bf);      // imported from a parked image, or captured under the option
    chain.push_back(std::move(plain));
    chain[0].used=1; chain[1].used=2;
    chain[0].pinned=true;     // the shared prefix every branch of this conversation mounts through
    {
        // a branch: the request resumes at the shorter checkpoint, the older ones stay in the chain, and
        // retention then rotates the chain down to its cap. Every survivor keeps the codec it was held in.
        std::vector<ConversationCheckpoint> branched;
        for (const auto& c:chain) if (c.ids.size()<=ids.size()) branched.push_back(c);
        check(branched.size()==2 && branched[0].state_codec==(uint8_t) StateCodec::Bf16 &&
              branched[1].state_codec==0 && branched[0].gdn==kept_gdn,
              "capture: branching off a chain keeps each checkpoint's own coding and its own bytes");
        branched.erase(branched.begin()+1);   // retention's victim: the cap is one, the pin stays
        check(branched.size()==1 && branched[0].pinned && branched[0].gdn==kept_gdn &&
              conversation_checkpoint_restore(branched[0],ss,g,err),
              "capture: the retained pin still restores after the rotation");
        ConversationCheckpoint split_cp=branched[0];
        ConversationCheckpoint part;
        part.ids=ids;
        part.state_codec=(uint8_t) StateCodec::Bf16;   // a second stage's part, captured the same way
        check(conversation_checkpoint_save(part,ss,g,err),"capture: a second stage's part under the option");
        const std::vector<uint8_t> part_gdn=part.gdn;
        split_cp.stage_parts.push_back(std::move(part));
        auto apart=conversation_checkpoints_split(std::vector<ConversationCheckpoint>{split_cp},1);
        std::vector<ConversationCheckpoint> back;
        check(apart.parts[0].size()==1 && apart.parts[0][0].state_codec==(uint8_t) StateCodec::Bf16 &&
              apart.parts[0][0].gdn==part_gdn &&
              conversation_checkpoints_merge(std::move(apart),back) && back.size()==1 &&
              back[0].stage_parts[0].state_codec==(uint8_t) StateCodec::Bf16 &&
              back[0].stage_parts[0].gdn==part_gdn && back[0].gdn==kept_gdn,
              "capture: a stage part captured under the option keeps its own codec through a split and a merge");
        check(conversation_checkpoint_restore(back[0].stage_parts[0],ss,g,err),
              "capture: and it restores into its stage's fp32 buffers");
    }
    {
        // re-parked and saved, both write codecs, with the chain holding a coded and an fp32 snapshot at once
        const ConversationView fp32_view{ids,images,chain,true};
        const ConversationView bf16_view{ids,images,chain,true,StateCodec::Bf16};
        SavedConversation as_fp32,as_bf16;
        check(conversation_snapshot_save(as_fp32,fp32_view,ss,g,draft.state,err),
              "capture: re-parking the chain with the engine configured fp32");
        check(conversation_snapshot_save(as_bf16,bf16_view,ss,g,draft.state,err),
              "capture: and with it configured bf16");
        check(as_fp32.checkpoints[0].state_codec==0 && as_fp32.checkpoints[0].gdn.size()==ref.gdn.size() &&
              as_fp32.checkpoints[1].state_codec==0,
              "capture: an fp32 write widens the coded snapshot it is given and leaves the fp32 one alone");
        check(as_bf16.checkpoints[0].state_codec==(uint8_t) StateCodec::Bf16 &&
              as_bf16.checkpoints[0].gdn==kept_gdn &&      // copied, never coded twice
              as_bf16.checkpoints[1].state_codec==(uint8_t) StateCodec::Bf16 &&
              as_bf16.checkpoints[1].gdn.size()==ref.gdn.size()/2,
              "capture: a bf16 write copies what is already bf16 byte for byte and codes what is not");
        check(conversation_snapshot_restore(as_bf16,ss,g,draft.state,err)==ConversationRestore::restored,
              "capture: and the re-parked chain restores");
        // the SAVE: the file's own representation is the configured one, whichever checkpoint it selects
        SessionSaveLive sl; sl.state_bytes=sizes.gdn+sizes.ple; sl.tokens=(uint64_t) ids.size();
        sl.images=0; sl.kv_layers=1;
        std::vector<ConversationCheckpoint> disk; std::string why;
        const auto agree=[&](uint64_t,std::string&){ return true; };
        check(session_save_checkpoints(chain,sl,StateCodec::Bf16,agree,disk,why) && disk.size()==1 &&
              disk[0].state_codec==(uint8_t) StateCodec::Bf16 && disk[0].gdn==kept_gdn,
              "capture: the SAVE's checkpoint list carries the selected snapshot as the chain holds it");
        check(session_save_peak_bytes(&disk[0],sl,StateCodec::Bf16) <=
                  session_save_peak_bytes(&disk[0],sl,StateCodec::Fp32) &&
              session_save_peak_bytes(&disk[0],sl,StateCodec::Bf16) <
                  session_save_peak_bytes(&disk[0],sl,StateCodec::Fp32),
              "capture: its preflight prices a coded file at the coded size and an fp32 one at the widening too");
        const ConversationView disk_view{ids,images,disk,true,StateCodec::Bf16};
        SavedConversation dmeta; std::vector<SessionKvSource> dsrc;
        check(conversation_snapshot_sources(dmeta,dsrc,disk_view,ss,g,draft.state,err) &&
              dmeta.checkpoints[0].state_codec==(uint8_t) StateCodec::Bf16 &&
              dmeta.checkpoints[0].gdn==kept_gdn && dmeta.live.state_codec==(uint8_t) StateCodec::Bf16,
              "capture: a bf16 SAVE writes the coded bytes the chain already holds");
        namespace fs=std::filesystem;
        const fs::path dir=fs::temp_directory_path()/("strata-capture-"+std::to_string(experts)+"-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
        const SessionFileIdentity id{1,2};
        auto version_of=[](const std::string& p){ std::ifstream f(p,std::ios::binary);
            std::vector<char> b(12); f.read(b.data(),12); return (uint8_t) b[8]; };
        size_t n1=0,n2=0,n3=0;
        const std::string p1=(dir/"old.bin").string(),p2=(dir/"new.bin").string(),p3=(dir/"coded.bin").string();
        SavedConversation old_meta,new_meta; std::vector<SessionKvSource> old_src,new_src;
        check(conversation_snapshot_sources(old_meta,old_src,fp32_view,ss,g,draft.state,err) &&
              session_file_write(p1,old_meta,old_src,id,n1,err) && version_of(p1)==1,
              "capture: a chain captured with the option off still writes a version 1 file");
        check(conversation_snapshot_sources(new_meta,new_src,fp32_view,ss,g,draft.state,err) &&
              session_file_write(p2,new_meta,new_src,id,n2,err) && n2==n1 && version_of(p2)==1,
              "capture: and it is byte-for-byte the same size, written twice from the same chain");
        std::ifstream f1(p1,std::ios::binary), f2(p2,std::ios::binary);
        const std::vector<uint8_t> b1((std::istreambuf_iterator<char>(f1)),std::istreambuf_iterator<char>()),
                               b2((std::istreambuf_iterator<char>(f2)),std::istreambuf_iterator<char>());
        check(b1==b2,"capture: the default path is byte-identical, twice over");
        check(session_file_write(p3,as_bf16,id,n3,err) && version_of(p3)==2 && n3<n1,
              "capture: a chain captured under the option writes a smaller version 2 file");
        SavedConversation r;
        size_t n4=0;
        check(session_file_read(p3,id,r,n4,err) && r.checkpoints[0].state_codec==(uint8_t) StateCodec::Bf16 &&
              r.checkpoints[0].gdn==kept_gdn && r.checkpoints[1].state_codec==(uint8_t) StateCodec::Bf16 &&
              conversation_snapshot_restore(r,ss,g,draft.state,err)==ConversationRestore::restored,
              "capture: and it reads back coded and restores");
        std::error_code ec; fs::remove_all(dir,ec);
    }
    // ---- 6. a capture that cannot allocate. `conversation_checkpoint_save` builds beside the checkpoint and
    // reports a refusal, so the checkpoint it was given and the session keep their bytes; the refusal itself is
    // built without allocating, because this is the path where there is nothing to allocate. The deterministic
    // version of that (an allocator armed to refuse a given size, over host-backed state) is in
    // conversation_capture_codec_test.cpp; what is checked here is that a chain holding a freshly captured coded
    // snapshot re-parks under a real address-space limit as a refusal and not a throw, with the chain intact.
#if defined(__linux__)
    {
        std::vector<ConversationCheckpoint> fat;
        fat.push_back(bf);                       // the snapshot this session captured under the option
        fat.push_back(ref);                      // and one it captured with the option off
        fat[0].gdn.assign(256u<<20, 0x51);       // big enough that converting the pair cannot fit in the child
        fat[0].state_codec=(uint8_t) StateCodec::Bf16;
        fat[1].gdn.assign(256u<<20, 0x61);
        const std::vector<uint8_t> was0=fat[0].gdn, was1=fat[1].gdn;
        const pid_t pid=::fork();
        if (pid==0) {
            long pages=0, rss=0;
            if (FILE* fp=std::fopen("/proc/self/statm","r")) { if (std::fscanf(fp,"%ld %ld",&pages,&rss)!=2) pages=0; std::fclose(fp); }
            rlimit rl {};
            rl.rlim_cur=rl.rlim_max=(rlim_t) pages*(rlim_t) ::sysconf(_SC_PAGESIZE) + (64u<<20);
            if (pages<=0 || ::setrlimit(RLIMIT_AS,&rl)!=0) ::_exit(10);
            std::vector<ConversationCheckpoint> out; std::string e; bool ok=false;
            try { ok=conversation_checkpoints_store(fat,StateCodec::Bf16,out,e); } catch (...) { ::_exit(11); }
            if (ok) ::_exit(12);                 // the limit did not bite: proves nothing
            ::_exit(!out.empty() ? 13 :
                    fat[0].gdn==was0 && fat[0].state_codec==(uint8_t) StateCodec::Bf16 &&
                    fat[1].gdn==was1 && fat[1].state_codec==0 && !e.empty() ? 0 : 14);
        }
        int status=0;
        check(pid>0 && ::waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0,
              "capture: a chain of snapshots captured under the option re-parks under pressure as a refusal, intact");
    }
#endif
}

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    for (int fmt : std::array<int,4>{kKvF16,kKvInt8,kKvQ4,3}) for (int mode : {0,1,2}) {
        if (fmt==3 && mode!=0) continue;
        Fixture f(fmt,mode);
        if (fmt==3) {
            std::string err;
            ConversationKv invalid;
            for (int unsupported : {1,2}) {
                f.state.kv_mode=unsupported;
                check(conversation_kv_bytes(f.state,f.g,9,true)==0,"hybrid streaming/ring rejected without block-mover abort");
                check(!conversation_kv_save(invalid,f.state,f.g,9,true,err),"hybrid streaming/ring capture fails closed");
            }
            f.state.kv_mode=0;
        }
        {
            std::string err;
            ConversationKv invalid;
            for (const int64_t bad : {-1LL, 97LL, (long long) std::numeric_limits<int64_t>::max()}) {
                check(!conversation_kv_save(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid save extent before allocation");
                check(!conversation_kv_restore(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid restore extent before copying");
                check(conversation_kv_bytes(f.state,f.g,bad,mode!=2)==0,"invalid extent cannot overflow byte estimate");
            }
        }
        for (int64_t upto : {0,1,3,4,5,63,64,65,96}) {
            const bool index=mode!=2;
            std::string err;
            f.fill(13);
            ConversationKv a,b,restored;
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(a,f.state,f.g,upto,index,err),"save A");
            check(a.bytes()==conversation_kv_bytes(f.state,f.g,upto,index),"size estimate equals snapshot payload");
            f.fill(177);
            check(conversation_kv_save(b,f.state,f.g,upto,index,err),"save B");
            check(upto==0 || !equal(a,b),"fixture changes state");
            check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"restore A over B");
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(restored,f.state,f.g,upto,index,err),"read back restored A");
            check(equal(a,restored),"A/B/A byte-exact K/V and indexer state");
            uint64_t fingerprint = 0;
            check(conversation_kv_verify(a,f.state,f.g,upto,index,fingerprint,err),"verify restored authoritative and resident bytes without rewriting them");
            {
                // disk save streams the same bytes as a captured image, without the host copy
                SessionKvSource src;
                check(conversation_kv_source(src,f.state,f.g,upto,index,err),"K/V source for a disk save");
                check(src.format==a.format && src.cells==a.cells && src.heads==a.heads && src.head_dim==a.head_dim &&
                      src.page_size==a.page_size && src.pooled_rows==a.pooled_rows && src.idx_dim==a.idx_dim,
                      "K/V source metadata equals the captured image");
                const std::array<const ConversationBuffer*,5> parts={&a.k,&a.v,&a.k_scale,&a.v_scale,&a.pooled};
                for (size_t i=0;i<5;++i) {
                    check(src.sizes[i]==parts[i]->size(),"K/V source part size");
                    std::vector<uint8_t> got(src.sizes[i]), want(src.sizes[i]);
                    const size_t half=src.sizes[i]/2;   // two reads: offsets are honoured
                    check(src.read(i,0,got.data(),half) && src.read(i,half,got.data()+half,src.sizes[i]-half),
                          "K/V source reads");
                    check(parts[i]->read(want.data(),0,want.size()) && got==want,"K/V source bytes equal the image");
                }
                SessionKvSource none;
                check(!conversation_kv_source(none,f.state,f.g,-1,index,err),"K/V source rejects an invalid extent");
            }
            if (mode==1) {
                std::vector<int32_t> table((size_t)f.state.n_pages);
                cuda_check(cudaMemcpy(table.data(),f.state.page_table,table.size()*4,cudaMemcpyDeviceToHost));
                check(std::all_of(table.begin(),table.end(),[](int32_t x){return x==-1;}),"stale VRAM pages invalidated");
            }
            if (mode==2 && upto>0) {
                const auto s=qsa_real_shapes();
                const int64_t b1=(upto+s.page_size-1)/s.page_size,b0=std::max<int64_t>(0,b1-f.state.n_slots);
                const size_t block=kv_block_bytes(s,fmt)/2;
                // Codes and scales are separate in INT8; check code bytes here.
                const size_t code_block=fmt==kKvInt8 ? s.page_size*s.n_head_kv*s.head_dim : block;
                const void* slots=fmt==kKvQ4?(void*)f.state.k_q4:fmt==kKvInt8?(void*)f.state.k_q:(void*)f.state.k_pool;
                std::vector<uint8_t> got(code_block);
                for (int64_t page=b0;page<b1;++page) {
                    cuda_check(cudaMemcpy(got.data(),(const uint8_t*)slots+(page%f.state.n_slots)*code_block,code_block,cudaMemcpyDeviceToHost));
                    std::vector<uint8_t> expected(code_block);
                    check(a.k.read(expected.data(), page*code_block, code_block),"read segmented expected draft page");
                    check(got == expected,"draft ring contains restored pages");
                }
            }
            auto bad=a; bad.head_dim++;
            check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject incompatible geometry");
            if (fmt==kKvInt8) {
                // #293: INT8 K/V saved through the Hadamard rotation (STRATA_KV_ROT=1) never restores into a state
                // without it, nor the reverse
                f.state.kv_rot=true;
                ConversationKv rotated;
                check(conversation_kv_save(rotated,f.state,f.g,upto,index,err),"save rotated INT8");
                check(!conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image into a rotated state rejected");
                f.state.kv_rot=false;
                check(!conversation_kv_restore(rotated,f.state,f.g,upto,index,err),"rotated image into an unrotated state rejected");
                check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image still restores");
            }
            if (!a.k.empty()) {
                bad=a; bad.k.pop_back();
                check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject malformed payload");
            }
        }
    }
    for (int fmt : {kKvF16,kKvInt8,kKvQ4}) for (int mode : {0,1}) for (int experts : {256,512})
        full_session(fmt,mode,experts);
    for (int experts : {256,512}) full_session(3,0,experts);
    for (int experts : {256,512}) fresh_capture(experts);
#if defined(__linux__)
    // ---- widening under memory pressure (review P2). The parked mount calls conversation_checkpoints_widen with
    // nothing around it, and widening allocates the fp32 array beside the coded one: a throw there ended the engine.
    // A child with its address space capped just above what it already holds is the honest way to make the
    // allocation fail without pretending to control the allocator from outside.
    {
        constexpr size_t kCoded = 256u << 20;                      // 256 MiB of bf16 -> 512 MiB widened
        ConversationCheckpoint c;                                   // built before the fork: the child must be able
        c.ids.assign(64, 7);                                        // to fail at the widening and nowhere else
        c.gdn.assign(kCoded, 0x11); c.ple = {1}; c.tails = {2}; c.dead = {3}; c.block_pos = {4};
        c.state_codec = (uint8_t) StateCodec::Bf16;
        std::vector<ConversationCheckpoint> chain{std::move(c)};
        const pid_t pid = ::fork();
        if (pid == 0) {
            long pages = 0, rss = 0;
            if (FILE* f = std::fopen("/proc/self/statm", "r")) { if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) pages = 0; std::fclose(f); }
            rlimit rl {};
            rl.rlim_cur = rl.rlim_max = (rlim_t) pages * (rlim_t) ::sysconf(_SC_PAGESIZE) + (64u << 20);
            if (pages <= 0 || ::setrlimit(RLIMIT_AS, &rl) != 0) ::_exit(10);
            std::string err; bool ok = false;
            try { ok = conversation_checkpoints_widen(chain, err); } catch (...) { ::_exit(11); }
            if (ok) ::_exit(12);                                    // the limit did not bite: proves nothing
            ::_exit(chain[0].state_codec == (uint8_t) StateCodec::Bf16 && chain[0].gdn.size() == kCoded ? 0 : 13);
        }
        int status = 0;
        check(pid > 0 && ::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "a widen that cannot allocate is refused to its caller, not thrown past it");
    }
    // ---- re-parking a MIXED chain under memory pressure. A resumed conversation now keeps the snapshots it
    // imported in the codec they were parked in, beside the fp32 ones it captured itself, so the next park converts
    // every one of them to the codec this engine writes: the copy allocates the whole stored image again. That is
    // the allocation the lazy chain puts on the park path, and it is a refusal, not a throw, with the chain the
    // engine is still running on left exactly as it was.
    {
        constexpr size_t kFp32 = 256u << 20;              // an fp32 snapshot this session captured
        ConversationCheckpoint fp, coded;
        fp.ids.assign(64, 5); fp.gdn.assign(kFp32, 0x31);
        coded.ids.assign(64, 5); coded.gdn.assign(kFp32 / 2, 0x41);   // one imported from a parked image
        coded.state_codec = (uint8_t) StateCodec::Bf16;
        std::vector<ConversationCheckpoint> chain; chain.push_back(std::move(fp)); chain.push_back(std::move(coded));
        const pid_t pid = ::fork();
        if (pid == 0) {
            long pages = 0, rss = 0;
            if (FILE* f = std::fopen("/proc/self/statm", "r")) { if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) pages = 0; std::fclose(f); }
            rlimit rl {};
            rl.rlim_cur = rl.rlim_max = (rlim_t) pages * (rlim_t) ::sysconf(_SC_PAGESIZE) + (64u << 20);
            if (pages <= 0 || ::setrlimit(RLIMIT_AS, &rl) != 0) ::_exit(10);
            std::vector<ConversationCheckpoint> out;
            std::string err; bool ok = false;
            try { ok = conversation_checkpoints_store(chain, StateCodec::Bf16, out, err); } catch (...) { ::_exit(11); }
            if (ok) ::_exit(12);                                    // the limit did not bite: proves nothing
            ::_exit(chain[0].gdn.size() == kFp32 && chain[0].state_codec == 0 &&
                    chain[1].gdn.size() == kFp32 / 2 && chain[1].state_codec == (uint8_t) StateCodec::Bf16 &&
                    !err.empty() ? 0 : 13);
        }
        int status = 0;
        check(pid > 0 && ::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "a park of a mixed chain that cannot allocate is refused, and leaves the active chain as it was");
    }
#endif
    std::printf("conversation_snapshot_test: %d checks passed\n",checks);
}
