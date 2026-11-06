#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/kv_pool.hpp"
#include "strata/kernels/kv_q4.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <cstdlib>
#include <limits>
#include <utility>
#include <memory>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {
// STRATA_CKPT_ASYNC=1 (the ctest entries conversation_snapshot_test_ckpt_async / _vram_slots): the checkpoint's GDN bytes land in
// the background; the test reads them at once, so it joins after each save (the engine's readers join before they read)
template <class... A> bool ckpt_save(A&&... a) {
    const bool ok = conversation_checkpoint_save(std::forward<A>(a)...);
    conversation_checkpoint_join(nullptr);
    return ok;
}
template <class... A> bool snap_save(A&&... a) {
    const bool ok = conversation_snapshot_save(std::forward<A>(a)...);
    conversation_checkpoint_join(nullptr);
    return ok;
}
}  // namespace

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
    Fixture(int fmt, int mode, int64_t cells = 96) {
        if (cells != 96) { g.n_head_kv=1; g.head_dim=64; g.idx_key_dim=8; }
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt==kKvInt8; st.kv_q4 = fmt==kKvQ4; st.kv_hybrid = fmt==3;
        st.n_pages = cells/4; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = mode==2 ? 2 : st.max_cells/4+2;
        const size_t per = fmt==kKvQ4 ? kv_q4_bytes_per_head((int)g.head_dim) : g.head_dim*((st.kv_int8 || st.kv_hybrid) ? 1:2);
        const size_t rows = st.max_cells*g.n_head_kv, slot_rows = st.n_slots*4*g.n_head_kv;
        sizes = {rows*per, rows*per, fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0,
                 fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0, (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
        if (fmt==3) {
            sizes = {rows*per, rows*kv_q4_bytes_per_head((int)g.head_dim), rows*(g.head_dim/64)*2, 0,
                     (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
            alloc(st.k_q,slot_rows*per);
            alloc(st.v_q4,slot_rows*kv_q4_bytes_per_head((int)g.head_dim));
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q4:st.v_q4;
            sources[2]=mode?st.host.k_scale:st.k_scale;
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
    // --kv-pool-tokens: the host copy of a streamed state lives in chunks of a pool shared with other states: block b of
    // this state is block (table[b >> 2] << 2 | b & 3) of the arrays (KvHostPools::chunk_host)
    static constexpr int kLanes = 3, kShift = 2;
    std::vector<int32_t> table;
    void make_pooled(Fixture* owner, int lane) {
        auto& h = state.host;
        if (owner == nullptr) {   // the pool: kLanes states' worth of blocks, in every host array
            const size_t n = size_t(kLanes);
            if (h.k_pool) { alloc(h.k_pool, sizes[0]*n, true); alloc(h.v_pool, sizes[1]*n, true); }
            if (h.k_q4) { alloc(h.k_q4, sizes[0]*n, true); alloc(h.v_q4, sizes[1]*n, true); }
            if (h.k_q) {
                alloc(h.k_q, sizes[0]*n, true);
                if (state.kv_hybrid) alloc(h.v_q4, sizes[1]*n, true);
                else alloc(h.v_q, sizes[1]*n, true);
                alloc(h.k_scale, sizes[2]*n, true); alloc(h.v_scale, sizes[3]*n, true);
            }
        } else {
            h = owner->state.host;
        }
        if (h.k_pool) { sources[0] = h.k_pool; sources[1] = h.v_pool; }
        if (h.k_q4) { sources[0] = h.k_q4; sources[1] = h.v_q4; }
        if (h.k_q) {
            sources[0] = h.k_q; sources[1] = state.kv_hybrid ? static_cast<void*>(h.v_q4) : h.v_q;
            sources[2] = h.k_scale; sources[3] = h.v_scale;
        }
        table.resize(size_t(state.n_pages) >> kShift);
        for (size_t c = 0; c < table.size(); ++c) table[c] = int32_t(c * kLanes + size_t(lane));   // interleaved: no two chunks adjacent
        h.chunk_host = table.data();
        h.chunk_shift = kShift;
    }
    // bytes [off, off + n) of array i of THIS state, written where its chunk table puts them
    void put(size_t i, size_t off, const uint8_t* p, size_t n) {
        if (table.empty() || i >= 4) { cuda_check(cudaMemcpy(static_cast<uint8_t*>(sources[i])+off,p,n,cudaMemcpyDefault)); return; }
        const size_t block = sizes[i] / size_t(state.n_pages);
        for (size_t done = 0; done < n;) {
            const size_t pos = off + done, b = pos / block, in = pos % block;
            const size_t len = std::min(block - in, n - done);
            const size_t phys = (size_t(table[b >> kShift]) << kShift | (b & ((1u << kShift) - 1))) * block + in;
            cuda_check(cudaMemcpy(static_cast<uint8_t*>(sources[i])+phys,p+done,len,cudaMemcpyDefault));
            done += len;
        }
    }
    void fill(uint8_t salt) {
        for (size_t i=0;i<sources.size();++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j=0;j<data.size();++j) data[j]=(uint8_t)(salt+i*31+j*7+j/257);
            if (!data.empty()) put(i,0,data.data(),data.size());
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
    if (std::getenv("STRATA_TEST_BIG_GDN")) { g.ssm_state_size=64; g.ssm_v_heads=64; }   // a GDN state over 1 MiB: the async checkpoint path
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
    check(ckpt_save(checkpoint,ss,g,err),"save complete running checkpoint");
    checkpoint.used = 17;
    checkpoints.push_back(std::move(checkpoint));
    const ConversationView view{ids,images,checkpoints,true};
    SavedConversation a,b,restored;
    check(snap_save(a,view,ss,g,draft.state,err),"capture complete A");
    check(a.checkpoints[0].used == 17,"upstream checkpoint LRU stamp survives capture");
    fill(177);
    check(snap_save(b,view,ss,g,draft.state,err),"capture complete B");
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
    check(snap_save(restored,view,ss,g,draft.state,err),"capture B after refused restore");
    check(restored.live.gdn==b.live.gdn && restored.live.dead==b.live.dead &&
          equal(restored.kv[0],b.kv[0]) && equal(restored.kv[1],b.kv[1]),"refusal preserves all tested state");
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore complete A");
    check(snap_save(restored,view,ss,g,draft.state,err),"capture restored A");
    check(restored.live.gdn==a.live.gdn && restored.live.ple==a.live.ple && restored.live.tails==a.live.tails &&
          restored.live.dead==a.live.dead && restored.live.block_pos==a.live.block_pos &&
          equal(restored.kv[0],a.kv[0]) && equal(restored.kv[1],a.kv[1]),"whole-session A/B/A exactness including spare key");
    check(ss.ple_prev[0]==64 && ss.ple_prev[1]==65,"PLE token window reconstructed");
    {
        // #2409: a restore writes into the session's own buffers. After the copy stream exists (the restores above made
        // it) a restore of either image allocates nothing on the card: free memory does not move (2 MiB allocation
        // granules; 4 MiB of slack for a desktop card's other programs).
        check(conversation_copy_stream_ready(0),"this thread has its copy stream after the first restore");
        cudaDeviceSynchronize();
        size_t before=0,after=0,total=0;
        check(cudaMemGetInfo(&before,&total)==cudaSuccess,"free memory before the repeated restores");
        for(int i=0;i<6;++i)
            check(conversation_snapshot_restore(i%2?b:a,ss,g,draft.state,err)==ConversationRestore::restored,"repeated restore");
        cudaDeviceSynchronize();
        check(cudaMemGetInfo(&after,&total)==cudaSuccess,"free memory after the repeated restores");
        check((long long)after+(4ll<<20)>=(long long)before,"six more restores allocate nothing on the card");
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"A is back");
    }
    {
        // a borrowed prefix (ConversationCache::borrows): the K/V of A's first tokens and A's checkpoint there, A
        // untouched - the same bytes as restoring A whole and rewinding to that checkpoint
        const int64_t at = (int64_t) a.checkpoints[0].ids.size();
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A whole");
        check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"rewind it to its checkpoint");
        ConversationKv want_main,want_draft,got_main,got_draft;
        ConversationCheckpoint want_state,got_state;
        want_state.ids=got_state.ids=a.checkpoints[0].ids;
        check(conversation_kv_save(want_main,main.state,g,at,true,err) &&
              conversation_kv_save(want_draft,draft.state,g,at,false,err) &&
              ckpt_save(want_state,ss,g,err),"capture the rewound prefix");
        fill(201);   // another conversation in the session
        const SavedConversation untouched=a;
        check(conversation_snapshot_restore_prefix(a,(int64_t)a.live.ids.size()+1,ss,g,draft.state,err)==
                  ConversationRestore::invalid &&
              conversation_snapshot_restore_prefix(a,0,ss,g,draft.state,err)==ConversationRestore::invalid,
              "a prefix outside the image is refused");
        check(conversation_snapshot_restore_prefix(a,at,ss,g,draft.state,err)==ConversationRestore::restored,
              "restore A's prefix only");
        check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"mount the checkpoint over it");
        check(conversation_kv_save(got_main,main.state,g,at,true,err) &&
              conversation_kv_save(got_draft,draft.state,g,at,false,err) &&
              ckpt_save(got_state,ss,g,err),"capture the borrowed prefix");
        check(equal(got_main,want_main) && equal(got_draft,want_draft),
              "borrowed K/V equals the whole restore's prefix");
        check(got_state.gdn==want_state.gdn && got_state.ple==want_state.ple && got_state.tails==want_state.tails &&
              got_state.dead==want_state.dead && got_state.block_pos==want_state.block_pos,"the same running state");
        check(untouched.live.ids==a.live.ids && equal(untouched.kv[0],a.kv[0]) && equal(untouched.kv[1],a.kv[1]),
              "the parked image is unchanged");
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
        check(snap_save(fresh,view,ss,g,draft.state,err),"full capture reference after growth or rewind");
        size_t peak=0,reused=0;
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.state,peak,err),"admit incremental capture peak");
        check(snap_save(incremental,view,ss,g,draft.state,err,std::move(reuse),&reused),"capture with retained pages");
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
        check(snap_save(stage,view,ss,g,nullptr,err),"capture a stage image");
        check(stage.kv.size()==a.kv.size()-1 && equal(stage.kv[0],a.kv[0]) && stage.live.gdn==a.live.gdn,
              "a stage image holds the session's own layers, no draft");
        check(stage.bytes()<a.bytes(),"a stage image is smaller by the draft ring");
        check(conversation_snapshot_validate(stage,ss,g,nullptr,err),"validate a stage image without a draft");
        check(!conversation_snapshot_validate(stage,ss,g,draft.state,err),"a stage image is refused where a draft is expected");
        check(!conversation_snapshot_validate(a,ss,g,nullptr,err),"an image with a draft is refused where none is expected");
        fill(201);
        check(snap_save(stage_b,view,ss,g,nullptr,err),"capture another stage state");
        check(conversation_snapshot_restore(a,ss,g,nullptr,err)==ConversationRestore::invalid,
              "restoring a draft image as a stage image is refused before any write");
        check(snap_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage_b.live.gdn &&
              equal(stage_back.kv[0],stage_b.kv[0]),"the refusal left the stage untouched");
        check(conversation_snapshot_restore(stage,ss,g,nullptr,err)==ConversationRestore::restored,"restore a stage image");
        check(snap_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage.live.gdn &&
              stage_back.live.dead==stage.live.dead && equal(stage_back.kv[0],stage.kv[0]),"stage A/B/A exactness");
        // its retained K/V: the next park copies only what changed, and equals a full capture
        ConversationKvReuse reuse{stage.kv,65,65,{}};
        SavedConversation fresh,incremental;
        size_t peak=0,reused=0;
        check(snap_save(fresh,view,ss,g,nullptr,err),"full stage capture reference");
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,nullptr,peak,err),"admit an incremental stage capture");
        check(snap_save(incremental,view,ss,g,nullptr,err,std::move(reuse),&reused),"incremental stage capture");
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

// Small geometry exercises production allocation/ownership without four model-sized contexts.
void shared_pool_snapshot(int fmt) {
    constexpr int64_t C=4096, cells=2*C+5, capacity=3*C;
    std::array<std::unique_ptr<Fixture>,4> f;
    std::array<SessionState,4> sessions;
    for (size_t n=0;n<f.size();++n) {
        f[n]=std::make_unique<Fixture>(fmt,1,capacity);
        sessions[n].max_cells=capacity; sessions[n].qsa_alloc=1;
        sessions[n].qsa_states=&f[n]->state;
    }
    KvPool pool;
    pool.set_share_prefix(true);
    std::string error;
    check(pool.init(f[0]->g,{{&sessions[0],-1}},8*C,error),"initialize opt-in physical prefix pool");
    for (size_t n=0;n<f.size();++n) {
        check(pool.attach(sessions[n],error),"attach physical prefix lane");
        auto& st=f[n]->state;
        check(st.prefix_pool==&pool,"attach enables automatic snapshot prefix retention");
        auto& h=st.host;
        f[n]->sources={h.k_pool ? static_cast<void*>(h.k_pool) :
                      h.k_q ? static_cast<void*>(h.k_q) : static_cast<void*>(h.k_q4),
                      h.v_pool ? static_cast<void*>(h.v_pool) :
                      h.v_q ? static_cast<void*>(h.v_q) : static_cast<void*>(h.v_q4),
                      h.k_scale,h.v_scale,st.idx_pooled};
    }
    auto ids=[&](size_t lane) {
        const auto* p=f[lane]->state.host.chunk_host;
        return std::vector<int32_t>(p,p+3);
    };
    // Independent byte pattern in each physical array, including scales and the indexer's private rows.
    auto write=[&](size_t lane,int64_t first,int64_t end,uint8_t salt) {
        auto& x=*f[lane];
        for (size_t part=0;part<5;++part) {
            if (!x.sizes[part]) continue;
            if (part==4) {
                std::vector<uint8_t> data(x.sizes[part],salt);
                cuda_check(cudaMemcpy(x.sources[part],data.data(),data.size(),cudaMemcpyDefault));
                continue;
            }
            const size_t stride=x.sizes[part]/capacity;
            for (int64_t at=first;at<end;) {
                const int64_t stop=std::min(end,(at/C+1)*C);
                std::vector<uint8_t> data(size_t(stop-at)*stride);
                for (size_t j=0;j<data.size();++j) data[j]=uint8_t(salt+part*31+(size_t(at)*stride+j)*7);
                const size_t off=(size_t(x.state.host.chunk_host[at/C])*C+size_t(at%C))*stride;
                cuda_check(cudaMemcpy(static_cast<uint8_t*>(x.sources[part])+off,data.data(),data.size(),cudaMemcpyDefault));
                at=stop;
            }
        }
    };
    check(pool.prepare_write(sessions[0],0,cells),"reserve source prefix and partial tail");
    write(0,0,capacity,13);
    ConversationKv image,expected;
    // The legacy deep-copy path supplies an independent, immutable byte reference for every format.
    auto flat=f[0]->state; flat.prefix_pool=nullptr;
    check(conversation_kv_save(expected,flat,f[0]->g,cells,true,error),"capture independent full byte reference");
    const auto original=ids(0);
    const auto* table_address=f[0]->state.host.chunk;
    check(conversation_kv_save(image,f[0]->state,f[0]->g,cells,true,error),"capture actual pool snapshot");
    check(image.prefix && image.prefix->chunks()==std::vector<int32_t>(original.begin(),original.begin()+2),
          "snapshot retains actual completed physical chunk IDs");
    check(equal(image,expected),"aliased snapshot equals deep copy in all physical arrays");
    check(image.bytes()<expected.bytes(),"cache accounting excludes already-budgeted pinned prefix payload");
    const std::array<const ConversationBuffer*,4> parts{&image.k,&image.v,&image.k_scale,&image.v_scale};
    for (size_t part=0;part<parts.size();++part) {
        if (!parts[part]->size()) continue;
        const auto& host = f[0]->state.host;
        check(host.allocation_host && host.allocation_device, "pool retains the allocator's host/device mapping");
        const size_t array_offset = reinterpret_cast<uintptr_t>(f[0]->sources[part]) -
                                    reinterpret_cast<uintptr_t>(host.allocation_device);
        const uint8_t* host_array = host.allocation_host + array_offset;
        const size_t stride=f[0]->sizes[part]/capacity;
        for (size_t c=0;c<2;++c)
            check(parts[part]->visit(c*C*stride,1,[&](const uint8_t* p,size_t,size_t) {
                return p==host_array+size_t(original[c])*C*stride;
            }),"snapshot buffers alias native CPU pool chunk pointers");
        check(parts[part]->visit(2*C*stride,1,[&](const uint8_t* p,size_t,size_t) {
            return p!=host_array+size_t(original[2])*C*stride;
        }),"snapshot partial tail is private");
    }
    for (size_t lane=1;lane<4;++lane) {
        if (lane==1) {
            check(pool.share(sessions[0],sessions[lane],cells),"production share creates sibling");
            ConversationKv copied;
            check(conversation_kv_save(copied,f[lane]->state,f[0]->g,cells,false,error),
                  "capture direct-share bytes before snapshot restore");
            check(copied.k==expected.k && copied.v==expected.v && copied.k_scale==expected.k_scale &&
                  copied.v_scale==expected.v_scale,"direct share copies partial tail and retains every KV array");
        }
        // Lanes 2 and 3 are empty: their physical sharing must come from restore, not a prior share call.
        check(conversation_kv_restore(image,f[lane]->state,f[0]->g,cells,true,error),"restore automatically shares retained prefix");
        auto physical=ids(lane);
        check(physical[0]==original[0] && physical[1]==original[1] && physical[2]!=original[2],
              "siblings share two complete chunks and own their partial tail");
        for (size_t other=1;other<lane;++other)
            check(physical[2]!=ids(other)[2],"each sibling tail has a distinct physical chunk");
        check(f[lane]->state.idx_pooled!=f[0]->state.idx_pooled,"indexer rows are never pooled");
        uint64_t fingerprint=0;
        check(conversation_kv_verify(expected,f[lane]->state,f[0]->g,cells,true,fingerprint,error),
              "restored sibling has all original KV bytes and private index rows");
    }
    // A rewind followed by a speculative candidate window straddles the physical chunk boundary.
    const auto before=ids(1);
    check(pool.prepare_write(f[1]->state,C-2,C+7),"QsaState write barrier COWs every speculative-window chunk");
    check(ids(1)[0]!=before[0] && ids(1)[1]!=before[1] && ids(1)[2]==before[2],
          "rewind window detaches both touched shared chunks only");
    {
        uint64_t fingerprint=0;
        check(conversation_kv_verify(expected,f[1]->state,f[0]->g,cells,true,fingerprint,error),
              "COW copies every untouched KV byte, scale and private index row before writing");
    }
    write(1,C-2,C+7,201);
    check(pool.free_cells()==0,"fixture consumes exactly the available COW chunks");
    const auto victim=ids(2);
    check(!pool.prepare_write(sessions[2],0,2*C+1),"exhausted COW reservation fails");
    check(ids(2)==victim && ids(0)==original,"exhaustion preserves all physical identities");
    check(f[0]->state.host.chunk==table_address,"device chunk-table address remains stable");
    for (size_t lane : {size_t(0),size_t(2),size_t(3)}) {
        uint64_t fingerprint=0;
        check(conversation_kv_verify(expected,f[lane]->state,f[0]->g,cells,true,fingerprint,error),
              "rewound speculative write cannot corrupt siblings or their index rows");
    }
    check(equal(image,expected),"COW leaves retained cache bytes and partial tail unchanged");
    check(pool.prepare_write(sessions[0],2*C,cells),"original owner's private partial tail remains writable");
    write(0,2*C,cells,91);
    check(equal(image,expected),"snapshot's private partial tail survives source append/rewrite");
    for (size_t lane : {size_t(2),size_t(3)}) {
        uint64_t fingerprint=0;
        check(conversation_kv_verify(expected,f[lane]->state,f[0]->g,cells,true,fingerprint,error),
              "source tail rewrite does not change restored sibling tails or indexer rows");
    }
    {
        ConversationKv short_expected,short_got;
        // Branch 2 still holds the original bytes, so a deep-copy prefix supplies the byte oracle.
        auto source=f[2]->state; source.prefix_pool=nullptr;
        check(conversation_kv_save(short_expected,source,f[0]->g,C+3,true,error),"capture independent shorter prefix");
        check(conversation_kv_restore_prefix(image,f[1]->state,f[0]->g,cells,C+3,true,error),
              "restore a borrowed prefix ending inside the second shared chunk");
        check(ids(1)[0]==original[0] && ids(1)[1]!=original[1],
              "borrowed-prefix restore shares only completed chunks and privately copies its tail");
        check(conversation_kv_save(short_got,f[1]->state,f[0]->g,C+3,true,error) &&
              equal(short_got,short_expected),"borrowed prefix has exact KV, scales and private index bytes");
    }
    // The existing disk format expands shared segments, never serializes physical IDs or pointers.
    {
        SavedConversation shared,loaded;
        shared.live.ids.resize(cells,1); shared.kv.push_back(image);
        const auto path=std::filesystem::temp_directory_path()/
            ("strata-shared-kv-"+std::to_string(fmt)+"-"+
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
        const SessionFileIdentity identity{1,2};
        size_t written=0,read=0;
        check(session_file_write(path.string(),shared,identity,written,error),"serialize physical shared snapshot");
        check(session_file_read(path.string(),identity,loaded,read,error),"read expanded shared snapshot");
        check(written==read && loaded.kv.size()==1 && !loaded.kv[0].prefix && equal(loaded.kv[0],expected),
              "disk round trip expands all shared bytes without ownership metadata");
        std::error_code ec; std::filesystem::remove(path,ec);
    }
    {
        // Pure pool exhaustion must be rejected before a restore writes even its first layer.
        pool.shrink(sessions[2],2*C);
        Fixture blocker(fmt,1,capacity);
        SessionState held;
        held.max_cells=capacity; held.qsa_alloc=1; held.qsa_states=&blocker.state;
        check(pool.attach(held,error),"attach restore admission pressure lane");
        check(pool.reserve(held,pool.free_cells()) && pool.free_cells()==0,"fill available private-tail capacity");
        const auto before=ids(2);
        const auto version=pool.version();
        const auto refs=pool.reference_cells();
        check(!pool.can_restore(sessions[2],*image.prefix,cells,cells+8),
              "admission rejects a shared prefix whose private restore tail cannot fit");
        check(ids(2)==before && pool.version()==version && pool.reference_cells()==refs && equal(image,expected),
              "refused restore admission preserves destination mapping and retained cache ownership");
        pool.detach(held);
        check(pool.can_restore(sessions[2],*image.prefix,cells,cells+8),
              "restore admission succeeds after genuine physical capacity is reclaimed");
        check(conversation_kv_restore(image,f[2]->state,f[0]->g,cells,true,error),
              "admitted restore applies the same physical mapping plan");
        uint64_t fingerprint=0;
        check(conversation_kv_verify(expected,f[2]->state,f[0]->g,cells,true,fingerprint,error),
              "retry after capacity rejection restores exact authoritative bytes");
    }
    for (const auto& ss:sessions) pool.detach(ss);
    check(pool.free_cells()==pool.total_cells()-2*C,"snapshot retains final completed chunks after all session owners detach");
    image={};
    check(pool.free_cells()==pool.total_cells(),"last snapshot owner releases final physical chunks");
}
}

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    for (int fmt : std::array<int,4>{kKvF16,kKvInt8,kKvQ4,3}) for (int mode_id : {0,1,2,3}) {
        // mode_id 3: a streamed state whose host copy is a chunk of a pool shared with `other` (--kv-pool-tokens)
        const bool pooled = mode_id == 3;
        const int mode = pooled ? 1 : mode_id;
        if (fmt==3 && mode_id==2) continue;
        Fixture f(fmt,mode);
        std::unique_ptr<Fixture> other;
        if (pooled) {
            f.make_pooled(nullptr,0);
            other = std::make_unique<Fixture>(fmt,mode);
            other->make_pooled(&f,1);
        }
        if (fmt==3) {
            std::string err;
            ConversationKv invalid;
            f.state.kv_mode=2;
            check(conversation_kv_bytes(f.state,f.g,9,true)==0,"hybrid ring rejected without block-mover abort");
            check(!conversation_kv_save(invalid,f.state,f.g,9,true,err),"hybrid ring capture fails closed");
            f.state.kv_mode=mode;
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
            if (pooled) {
                // a second state in the same pool, its chunks interleaved with f's (--kv-pool-tokens): the direct copy
                // between two chunk tables, a neighbour's writes, and a prefix restore all go through the tables
                ConversationKv theirs, after, want, got;
                other->fill(99);
                check(conversation_kv_copy(f.state,other->state,f.g,upto,index,err,false),"direct copy between two pooled states");
                cuda_check(cudaDeviceSynchronize());
                check(conversation_kv_save(theirs,other->state,f.g,upto,index,err) && equal(a,theirs),
                      "the direct copy between two pooled states is byte-exact");
                other->fill(55);
                check(conversation_kv_save(after,f.state,f.g,upto,index,err) && equal(a,after),
                      "a neighbour's writes do not reach this state's chunks");
                if (upto == 65) {
                    Fixture resident(fmt, 0);
                    ConversationKv resident_image, pooled_image;
                    check(conversation_kv_copy(f.state,resident.state,f.g,upto,index,err,true),
                          "queued direct copy from interleaved host chunks to resident K/V");
                    cuda_check(cudaDeviceSynchronize());
                    check(conversation_kv_save(resident_image,resident.state,f.g,upto,index,err) && equal(a,resident_image),
                          "host-to-resident copy follows the source chunk map");
                    check(conversation_kv_copy(resident.state,other->state,f.g,upto,index,err,true),
                          "queued direct copy from resident K/V to interleaved host chunks");
                    cuda_check(cudaDeviceSynchronize());
                    check(conversation_kv_save(pooled_image,other->state,f.g,upto,index,err) && equal(a,pooled_image),
                          "resident-to-host copy follows the target chunk map");
                }
                const int64_t part = upto / 2;
                if (part > 0) {
                    check(conversation_kv_restore_prefix(a,other->state,f.g,upto,part,index,err),"prefix restore into a pooled state");
                    cuda_check(cudaDeviceSynchronize());
                    check(conversation_kv_save(want,f.state,f.g,part,index,err) && conversation_kv_save(got,other->state,f.g,part,index,err) &&
                          equal(want,got),"the prefix restored through the chunk table equals the source's prefix");
                }
            }
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
    for (int fmt : {kKvF16,kKvInt8,kKvQ4,kKvHybrid}) shared_pool_snapshot(fmt);
    std::printf("conversation_snapshot_test: %d checks passed\n",checks);
}
