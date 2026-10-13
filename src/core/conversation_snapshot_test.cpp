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
}
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
#endif
    std::printf("conversation_snapshot_test: %d checks passed\n",checks);
}
