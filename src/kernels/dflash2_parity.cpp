#include "strata/kernels/dflash2.hpp"
#include "strata/spec/dflash_policy.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void checked(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { checked(cudaMalloc(&p, n * sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { checked(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice)); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    std::vector<T> read(size_t n) { std::vector<T> v(n); checked(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
void near(float got, double want) {
    if (!std::isfinite(got) || std::abs(got - want) > 3e-5 * (1 + std::abs(want)))
        throw std::runtime_error("GPU/reference mismatch: " + std::to_string(got) + " vs " + std::to_string(want));
}
void convolve() {
    using namespace strata::kernels;
    const int hidden = 32;
    for (int width : {1, 2, 7, 8}) for (int taps : {1, 2, 5, 8}) for (int group : {1, 4, 16}) {
        const int rows = 2 * width, groups = hidden / group;
        std::vector<float> x(rows * hidden), dynamic(rows * 2 * taps * groups), base(2 * taps * hidden);
        for (size_t i = 0; i < x.size(); ++i) x[i] = ((int)(i % 31) - 15) * 0.13f;
        // The second block's radically different first row reveals cross-block leakage.
        for (int c = 0; c < hidden; ++c) x[width * hidden + c] = 10 + c;
        for (size_t i = 0; i < dynamic.size(); ++i) dynamic[i] = ((int)(i % 17) - 8) * 0.03f;
        for (size_t i = 0; i < base.size(); ++i) base[i] = ((int)(i % 13) - 6) * 0.02f;
        Device<float> dx(x), dd(dynamic), db(base), result(x.size());
        for (int side = 0; side < 2; ++side) {
            dflash2_convolve(dx.p, dd.p, db.p, result.p, rows, hidden, taps, group, side, width, nullptr);
            checked(cudaGetLastError());
            const auto got = result.read(x.size());
            for (int row = 0; row < rows; ++row) for (int c = 0; c < hidden; ++c) {
                double expected = 0;
                for (int offset = 0; offset < taps; ++offset) {
                    if (row % width < offset) continue;
                    const double weight = (double) base[(side * taps + offset) * hidden + c] +
                        dynamic[((row * 2 + side) * taps + offset) * groups + c / group];
                    expected += weight * x[(row - offset) * hidden + c];
                }
                near(got[row * hidden + c], expected);
            }
        }
    }
}
void selector(bool timing) {
    using namespace strata::kernels;
    const int vocab = 17;
    int reranked = 0;
    for (int rank : {1,3,256,1024})
    for (int rows : {2, 3, 8}) for (int top_k : {1, 3, 16, 17}) {
        std::vector<float> logits(rows * vocab), hidden(rows * rank), prev(vocab * rank), next(prev.size());
        for (size_t i = 0; i < logits.size(); ++i) logits[i] = ((int)(i % 7) - 3) * 0.05f + (i % vocab) * 0.00037f;
        std::fill(logits.begin(), logits.begin() + vocab, 0.1f); // top-k tie test
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = ((int)(i % 5) - 2) * 0.21f;
        for (size_t i = 0; i < prev.size(); ++i) { prev[i] = ((int)(i % 13) - 6) * 0.23f; next[i] = ((int)(i % 17) - 8) * 0.31f; }
        std::vector<uint16_t> p16(prev.size()),n16(next.size());
        for(size_t i=0;i<prev.size();++i) {
            uint32_t p,n;std::memcpy(&p,&prev[i],4);std::memcpy(&n,&next[i],4);
            p16[i]=(p+0x7fff+((p>>16)&1))>>16;n16[i]=(n+0x7fff+((n>>16)&1))>>16;
            p=(uint32_t)p16[i]<<16;n=(uint32_t)n16[i]<<16;
            std::memcpy(&prev[i],&p,4);std::memcpy(&next[i],&n,4);
        }
        Device<uint16_t> dp16(p16),dn16(n16);
        Device<float> dl(logits), dh(hidden), dp(prev), dn(next), du(rows * top_k);
        Device<int32_t> anchor(std::vector<int32_t>{7}), ids(rows * top_k), output(rows - 1);
        Device<float> confidence(rows - 1);
        dflash2_topk(dl.p, ids.p, du.p, rows, vocab, top_k, nullptr);
        checked(cudaGetLastError());
        auto got_ids = ids.read(rows * top_k);
        auto got_unary = du.read(rows * top_k);
        std::vector<int> order(vocab);
        for (int row = 0; row < rows; ++row) {
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[row * vocab + a] > logits[row * vocab + b]; });
            for (int k = 0; k < top_k; ++k) {
                if (got_ids[row * top_k + k] != order[k]) throw std::runtime_error("top-k IDs/tie order differ");
                near(got_unary[row * top_k + k], logits[row * vocab + order[k]]);
            }
        }
        for (bool probabilities : {false, true}) {
            dflash2_select(ids.p, du.p, dh.p, dp.p, dn.p, anchor.p, output.p,
                           probabilities ? confidence.p : nullptr, rows, top_k, rank, nullptr);
            checked(cudaGetLastError());
            auto got = output.read(rows - 1);
            auto prob = probabilities ? confidence.read(rows - 1) : std::vector<float>{};
            dflash2_select_bf16(ids.p,du.p,dh.p,dp16.p,dn16.p,anchor.p,output.p,
                                probabilities ? confidence.p : nullptr,rows,top_k,rank,nullptr);
            if(output.read(rows-1)!=got || (probabilities && confidence.read(rows-1)!=prob))
                throw std::runtime_error("BF16 and FP32 codebook selector results differ");
            dflash2_select_bf16(ids.p,du.p,dh.p,dp16.p,dn16.p,anchor.p,output.p,
                                probabilities ? confidence.p : nullptr,rows,top_k,rank,nullptr,true);
            if(output.read(rows-1)!=got || (probabilities && confidence.read(rows-1)!=prob))
                throw std::runtime_error("reference-mode BF16 selector results differ");
            if (timing && rows==8 && rank==256 && top_k==16) {
                cudaEvent_t a,b;checked(cudaEventCreate(&a));checked(cudaEventCreate(&b));
                for(int i=0;i<10;++i)dflash2_select_bf16(ids.p,du.p,dh.p,dp16.p,dn16.p,anchor.p,output.p,
                    probabilities?confidence.p:nullptr,rows,top_k,rank,nullptr);
                checked(cudaEventRecord(a));
                for(int i=0;i<100;++i)dflash2_select_bf16(ids.p,du.p,dh.p,dp16.p,dn16.p,anchor.p,output.p,
                    probabilities?confidence.p:nullptr,rows,top_k,rank,nullptr);
                checked(cudaEventRecord(b));checked(cudaEventSynchronize(b));float ms;
                checked(cudaEventElapsedTime(&ms,a,b));
                std::printf("BF16 selector rows=8 rank=256 k=16 confidence=%d: %.6f ms/pass\n",probabilities,ms/100);
                cudaEventDestroy(a);cudaEventDestroy(b);
            }
            // The runtime may omit every anchor-row head operation. Poison
            // that row and fill only the proposal slices; picks AND confidence
            // must stay bit-identical, including the smallest two-row block.
            checked(cudaMemset(ids.p, 0xff, top_k * sizeof(int32_t)));
            const std::vector<float> poison(std::max(top_k, rank), NAN);
            checked(cudaMemcpy(du.p, poison.data(), top_k * sizeof(float), cudaMemcpyHostToDevice));
            checked(cudaMemcpy(dh.p, poison.data(), rank * sizeof(float), cudaMemcpyHostToDevice));
            dflash2_topk(dl.p + vocab, ids.p + top_k, du.p + top_k,
                         rows - 1, vocab, top_k, nullptr);
            dflash2_select_bf16(ids.p, du.p, dh.p, dp16.p, dn16.p, anchor.p, output.p,
                               probabilities ? confidence.p : nullptr, rows, top_k, rank, nullptr);
            if (output.read(rows - 1) != got || (probabilities && confidence.read(rows - 1) != prob))
                throw std::runtime_error("omitting anchor logits changed the selector");
            int predecessor = 7;
            for (int row = 1; row < rows; ++row) {
                std::vector<double> scores(top_k);
                for (int k = 0; k < top_k; ++k) {
                    int token = got_ids[row * top_k + k];
                    double interaction = 0;
                    for (int j = 0; j < rank; ++j) interaction += (double) prev[predecessor * rank + j] * hidden[row * rank + j] * next[token * rank + j];
                    scores[k] = logits[row * vocab + token] + interaction;
                }
                const int best = (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
                predecessor = got_ids[row * top_k + best];
                if (got[row - 1] != predecessor) {
                    std::fprintf(stderr, "rows=%d topk=%d row=%d got=%d want=%d\n", rows, top_k, row, got[row - 1], predecessor);
                    for (int k = 0; k < top_k; ++k) std::fprintf(stderr, " candidate=%d score=%.12g\n", got_ids[row * top_k + k], scores[k]);
                    throw std::runtime_error("selector path differs");
                }
                reranked += best != 0;
                if (probabilities) {
                    double sum = 0;
                    for (double score : scores) sum += std::exp(score - scores[best]);
                    near(prob[row - 1], 1.0 / sum);
                }
            }
        }
        // Invalid logits must propagate failure, including every subsequent path position.
        checked(cudaMemset(dl.p, 0xff, logits.size() * sizeof(float))); // NaN
        dflash2_topk(dl.p, ids.p, du.p, rows, vocab, top_k, nullptr);
        dflash2_select(ids.p, du.p, dh.p, dp.p, dn.p, anchor.p, output.p, confidence.p, rows, top_k, rank, nullptr);
        for (auto id : output.read(rows - 1)) if (id != -1) throw std::runtime_error("nonfinite selector produced a token");
    }
    if (!reranked) throw std::runtime_error("fixture did not exercise selector reranking");
}
void target_features_and_attention() {
    using namespace strata::kernels;
    {
        const int rows = 3, hidden = 33, hc = 4;
        std::vector<float> r(rows*hidden*hc);
        for (size_t i = 0; i < r.size(); ++i) r[i] = ((int)(i%31)-15)*0.25f;
        Device<float> dr(r), out(rows*hidden); Device<uint16_t> bf(rows*hidden);
        dflash2_hc_mean(dr.p,out.p,bf.p,rows,hidden,hc,nullptr);
        auto got=out.read(rows*hidden); auto bits=bf.read(rows*hidden);
        for (int i=0;i<rows*hidden;++i) {
            float expected=0;
            for(int h=0;h<hc;++h) expected+=r[(i/hidden*hc+h)*hidden+i%hidden]/hc;
            near(got[i],expected);
            uint32_t u; std::memcpy(&u,&expected,4);
            if(bits[i] != ((u+0x7fff+((u>>16)&1))>>16)) throw std::runtime_error("HC BF16 mean differs");
        }
    }
    const int hd=128;
    for (int page : {64,96})
    for (int heads : {4,32}) for (int rows : {2,4,8})
    for (int end : {9,73,2053}) for (int window : {0,2048}) {
        const int kv = heads == 32 ? 8 : 2;
        const int pages=(end+page-1)/page;
        std::vector<int32_t> table(pages), steps(rows*4);
        for(int p=0;p<pages;++p)table[p]=pages-1-p; // nonidentity page table
        std::vector<uint16_t> k(pages*page*kv*hd),v(k.size());
        std::vector<float> q(rows*heads*hd);
        for(size_t i=0;i<k.size();++i) {
            k[i]=__half_as_ushort(__float2half(((int)(i%43)-21)*0.015f));
            v[i]=__half_as_ushort(__float2half(((int)(i%37)-18)*0.03f));
        }
        for(size_t i=0;i<q.size();++i)q[i]=((int)(i%29)-14)*0.02f;
        for(int r=0;r<rows;++r)steps[r*4+1]=end;
        const int visible = std::min(end, window ? window + rows : end);
        const int padded = (int) strata::spec::dflash_attention_bucket(visible, 8192);
        Device<float> dq(q), out(q.size()), scratch(rows*heads*((padded+63)/64)*130);
        Device<uint16_t> dk(k),dv(v);Device<int32_t> dt(table),ds(steps);
        dflash2_attention(dq.p,dk.p,dv.p,dt.p,ds.p,out.p,scratch.p,rows,heads,kv,page,
                          std::min(end,window?window+rows:end),window,nullptr);
        auto got=out.read(q.size());
        if (heads == 32 && rows == 2 && end == 2053 && window == 2048 && page == 64) {
            Device<uint16_t> shifted(v.size() + 1);
            checked(cudaMemcpy(shifted.p + 1, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
            dflash2_attention(dq.p,dk.p,shifted.p+1,dt.p,ds.p,out.p,scratch.p,rows,heads,kv,page,
                              visible,window,nullptr);
            if (out.read(q.size()) != got) throw std::runtime_error("unaligned V changed attention bits");
        }
        dflash2_attention(dq.p,dk.p,dv.p,dt.p,ds.p,out.p,scratch.p,rows,heads,kv,page,
                          padded,window,nullptr);
        if (out.read(q.size()) != got) throw std::runtime_error("padded DFlash2 attention changed output bits");
        for(int r=0;r<rows;++r)for(int h=0;h<heads;++h) {
            std::vector<double> scores(end,-INFINITY);double m=-INFINITY;
            for(int c=0;c<end;++c) {
                if(window && std::abs(c-(end-rows+r))>=window)continue;
                const size_t pr=((size_t)table[c/page]*kv+h/(heads/kv))*page+c%page;
                double s=0;
                for(int d=0;d<hd;++d)s+=q[(r*heads+h)*hd+d]*__half2float(__ushort_as_half(k[pr*hd+d]));
                scores[c]=s/std::sqrt((double)hd);m=std::max(m,scores[c]);
            }
            double sum=0;for(double& s:scores){s=std::exp(s-m);sum+=s;}
            for(int d=0;d<hd;++d) {
                double acc=0;
                for(int c=0;c<end;++c) {
                    const size_t pr=((size_t)table[c/page]*kv+h/(heads/kv))*page+c%page;
                    acc+=scores[c]*__half2float(__ushort_as_half(v[pr*hd+d]));
                }
                near(got[(r*heads+h)*hd+d],acc/sum);
            }
        }
    }
}

// Real drafter geometry, including a full local window. The output fingerprint
// and optional binary dump let before/after binaries compare their outputs,
// alongside the FP64 reference checks above. Timing uses graph replay.
void attention_benchmark(const char* bits_path) {
    using namespace strata::kernels;
    std::ofstream bits;
    if (bits_path) {
        bits.open(bits_path, std::ios::binary | std::ios::trunc);
        if (!bits) throw std::runtime_error("cannot open attention output dump");
    }
    for (int rows : {2, 4, 8}) for (int end : {32, 128, 512, 1024, 2053}) {
        const int heads = 32, kv = 8, page = 64, hd = 128;
        const int pages = (end + page - 1) / page;
        const int active = std::min(end, 2048 + rows);
        std::vector<int32_t> table(pages), steps(rows * 4);
        for (int p = 0; p < pages; ++p) table[p] = pages - 1 - p;
        for (int r = 0; r < rows; ++r) steps[r * 4 + 1] = end;
        std::vector<uint16_t> k(pages * page * kv * hd), v(k.size());
        std::vector<float> q(rows * heads * hd);
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = __half_as_ushort(__float2half(((int)(i % 43) - 21) * 0.015f));
            v[i] = __half_as_ushort(__float2half(((int)(i % 37) - 18) * 0.03f));
        }
        for (size_t i = 0; i < q.size(); ++i) q[i] = ((int)(i % 29) - 14) * 0.02f;
        Device<float> dq(q), out(q.size()), scratch(rows * heads * ((active + 63) / 64) * 130);
        Device<uint16_t> dk(k), dv(v);
        Device<int32_t> dt(table), ds(steps);
        cudaStream_t stream;
        checked(cudaStreamCreate(&stream));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        dflash2_attention(dq.p, dk.p, dv.p, dt.p, ds.p, out.p, scratch.p,
                          rows, heads, kv, page, active, 2048, stream);
        checked(cudaStreamEndCapture(stream, &graph));
        checked(cudaGraphInstantiate(&exec, graph, 0));
        for (int i = 0; i < 20; ++i) checked(cudaGraphLaunch(exec, stream));
        cudaEvent_t a, b;
        checked(cudaEventCreate(&a)); checked(cudaEventCreate(&b));
        checked(cudaEventRecord(a, stream));
        for (int i = 0; i < 200; ++i) checked(cudaGraphLaunch(exec, stream));
        checked(cudaEventRecord(b, stream)); checked(cudaEventSynchronize(b));
        float ms;
        checked(cudaEventElapsedTime(&ms, a, b));
        const auto got = out.read(q.size());
        if (bits_path) {
            bits.write(reinterpret_cast<const char*>(got.data()), got.size() * sizeof(float));
            if (!bits) throw std::runtime_error("cannot write attention output dump");
        }
        uint64_t fingerprint = 14695981039346656037ull;
        for (float value : got) {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof bits);
            fingerprint = (fingerprint ^ bits) * 1099511628211ull;
        }
        std::printf("attention rows=%d heads=32 end=%d: %.6f ms/pass bits=%016llx\n",
                    rows, end, ms / 200, (unsigned long long)fingerprint);
        checked(cudaEventDestroy(a)); checked(cudaEventDestroy(b));
        checked(cudaGraphExecDestroy(exec)); checked(cudaGraphDestroy(graph));
        checked(cudaStreamDestroy(stream));
    }
}

void full_vocabulary_topk(bool timing) {
    using namespace strata::kernels;
    const int rows=8,vocab=248320;
    std::vector<float> logits(rows*vocab);
    for(size_t i=0;i<logits.size();++i) logits[i]=std::sin((float)(i%10007)*0.017f);
    // Exact ties, boundary tokens, NaNs and infinities are present in real head diagnostics.
    logits[17]=NAN;logits[18]=INFINITY;logits[19]=-INFINITY;
    logits[1]=3;logits[vocab-1]=3;
    Device<float> dl(logits),du(rows*64);Device<int32_t> di(rows*64);
    for(int top:{1,16,64}) {
        dflash2_topk(dl.p,di.p,du.p,rows,vocab,top,nullptr);
        auto ids=di.read(rows*top);auto unary=du.read(rows*top);
        Device<uint8_t> workspace(dflash2_topk_scratch_bytes(rows,vocab,top));
        dflash2_topk(dl.p,di.p,du.p,rows,vocab,top,nullptr,workspace.p,
                     dflash2_topk_scratch_bytes(rows,vocab,top));
        auto tiled_ids=di.read(rows*top);auto tiled_unary=du.read(rows*top);
        if (tiled_ids!=ids || std::memcmp(tiled_unary.data(),unary.data(),unary.size()*sizeof(float)))
            throw std::runtime_error("tiled full-vocabulary top-k changed output bits");
        for(int r=0;r<rows;++r) {
            std::vector<int> order;
            for(int i=0;i<vocab;++i)if(std::isfinite(logits[r*vocab+i]))order.push_back(i);
            std::partial_sort(order.begin(),order.begin()+top,order.end(),[&](int a,int b){
                float x=logits[r*vocab+a],y=logits[r*vocab+b];return x>y || (x==y && a<b);
            });
            for(int k=0;k<top;++k) {
                if(ids[r*top+k]!=order[k] || unary[r*top+k]!=logits[r*vocab+order[k]])
                    throw std::runtime_error("full vocabulary top-k differs from CPU");
            }
        }
    }
    if(timing) {
        Device<uint8_t> workspace(dflash2_topk_scratch_bytes(rows,vocab,16));
        cudaEvent_t a,b;checked(cudaEventCreate(&a));checked(cudaEventCreate(&b));
        for (bool tiled : {false,true}) {
        void* scratch=tiled?workspace.p:nullptr;
        const size_t bytes=tiled?dflash2_topk_scratch_bytes(rows,vocab,16):0;
        for(int i=0;i<10;++i)dflash2_topk(dl.p,di.p,du.p,rows,vocab,16,nullptr,scratch,bytes);
        checked(cudaEventRecord(a));
        for(int i=0;i<100;++i)dflash2_topk(dl.p,di.p,du.p,rows,vocab,16,nullptr,scratch,bytes);
        checked(cudaEventRecord(b));checked(cudaEventSynchronize(b));float ms;
        checked(cudaEventElapsedTime(&ms,a,b));std::printf("topk rows=8 vocab=248320 k=16 %s: %.6f ms/pass\n",tiled?"tiled":"single CTA",ms/100);
        }
        cudaEventDestroy(a);cudaEventDestroy(b);
    }
}

void topk_tile_edges() {
    using namespace strata::kernels;
    for (int rows : {1,3,7}) for (int vocab : {4095,4096,4097,8193}) for (int top : {1,16,64}) {
        std::vector<float> logits(rows*vocab);
        for (size_t i=0;i<logits.size();++i) logits[i]=(int)(i%97)-48;
        // All-invalid rows, fewer finite values than K, exact ties (including
        // signed zero), and winners on both sides of a partition boundary.
        std::fill(logits.begin(),logits.begin()+vocab,NAN);
        logits[0]=-0.0f;logits[vocab-1]=0.0f;
        if (rows>1) std::fill(logits.begin()+vocab,logits.begin()+2*vocab,-INFINITY);
        Device<float> dl(logits),unary(rows*top);Device<int32_t> ids(rows*top);
        Device<uint8_t> workspace(dflash2_topk_scratch_bytes(rows,vocab,top));
        dflash2_topk(dl.p,ids.p,unary.p,rows,vocab,top,nullptr);
        const auto ref_ids=ids.read(rows*top);const auto ref_scores=unary.read(rows*top);
        dflash2_topk(dl.p,ids.p,unary.p,rows,vocab,top,nullptr,workspace.p,
                    dflash2_topk_scratch_bytes(rows,vocab,top));
        const auto scores=unary.read(rows*top);
        if (ids.read(rows*top)!=ref_ids || std::memcmp(scores.data(),ref_scores.data(),scores.size()*sizeof(float)))
            throw std::runtime_error("tiled top-k boundary/invalid/tie parity failed");
        bool refused=false;
        try {dflash2_topk(dl.p,ids.p,unary.p,rows,vocab,top,nullptr,workspace.p,1);}
        catch(const std::invalid_argument&) {refused=true;}
        if (!refused) throw std::runtime_error("top-k accepted undersized workspace");
    }
}

void invalid_model_values() {
    using namespace strata::kernels;
    // A visible NaN query must remain invalid, rather than turning into zero attention.
    const int rows=2,heads=1,kv=1,page=4;
    std::vector<float> q(rows*128,0.1f);q[0]=NAN;
    Device<float> dq(q),out(q.size()),scratch(rows*130);
    Device<uint16_t> dk(std::vector<uint16_t>(page*128,__half_as_ushort(__float2half(0.5f)))),dv(std::vector<uint16_t>(page*128,0));
    Device<int32_t> table(std::vector<int32_t>{0}),steps(std::vector<int32_t>{0,2,0,2,0,2,0,2});
    dflash2_attention(dq.p,dk.p,dv.p,table.p,steps.p,out.p,scratch.p,rows,heads,kv,page,2,0,nullptr);
    auto got=out.read(q.size());
    for(int d=0;d<128;++d)if(!std::isnan(got[d]))throw std::runtime_error("attention masked a visible NaN");
    // An invalid candidate transition must not yield a finite token with NaN confidence.
    Device<int32_t> ids(std::vector<int32_t>{0,1,0,1,0,1}),anchor(std::vector<int32_t>{0}),picks(2);
    Device<float> unary(std::vector<float>(6,0)),hidden(std::vector<float>(3,1)),
        prev(std::vector<float>{1,1}),next(std::vector<float>{1,NAN}),prob(2);
    dflash2_select(ids.p,unary.p,hidden.p,prev.p,next.p,anchor.p,picks.p,prob.p,3,2,1,nullptr);
    for(int t:picks.read(2))if(t!=-1)throw std::runtime_error("selector masked an invalid transition");
    for(float p:prob.read(2))if(p!=0)throw std::runtime_error("invalid selector confidence is not zero");
    Device<uint16_t> prev16(std::vector<uint16_t>{0x3f80,0x3f80}),
        next16(std::vector<uint16_t>{0x3f80,0x7fc0});
    dflash2_select_bf16(ids.p,unary.p,hidden.p,prev16.p,next16.p,anchor.p,picks.p,prob.p,3,2,1,nullptr);
    for(int t:picks.read(2))if(t!=-1)throw std::runtime_error("BF16 selector masked an invalid transition");
    Device<int32_t> negative_anchor(std::vector<int32_t>{-1});
    dflash2_select(ids.p,unary.p,hidden.p,prev.p,next.p,negative_anchor.p,picks.p,prob.p,3,2,1,nullptr);
    for(int t:picks.read(2))if(t!=-1)throw std::runtime_error("negative selector anchor accepted");
    // Rejected API calls must not launch a kernel which poisons the CUDA context.
    bool refused=false;
    try { dflash2_select(ids.p,unary.p,hidden.p,prev.p,next.p,anchor.p,picks.p,prob.p,3,2,1025,nullptr); }
    catch(const std::invalid_argument&) {refused=true;}
    if(!refused)throw std::runtime_error("FP32 selector accepts unsupported rank");
}

}
int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    const bool timing=argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    try { invalid_model_values(); convolve(); selector(timing); target_features_and_attention(); topk_tile_edges(); full_vocabulary_topk(timing); if (timing) attention_benchmark(argc > 2 ? argv[2] : nullptr); checked(cudaDeviceSynchronize()); }
    catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
    std::puts("PASS: DFlash2 convolution, block isolation, top-k, conditioned path, confidence and nonfinite refusal");
    return 0;
}
