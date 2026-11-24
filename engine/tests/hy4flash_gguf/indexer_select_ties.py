#!/usr/bin/env python3
"""Check the production CUDA selection kernel against a CPU sort, including cutoff ties.

Run with CUDA_VISIBLE_DEVICES set on a test machine. NVCC may name an absolute compiler path.
The kernel is extracted verbatim so this test needs neither model weights nor engine linking.
"""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
source = (root / "src/hy4flash_gguf/indexer.cu").read_text()
start = source.index("__global__ __launch_bounds__(kBlock) void select_k(")
end = source.index("\n}  // namespace", start)
kernel = source[start:end]
header = """
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>
constexpr int kBlock=256;
"""
main = r"""
int main() {
    int cases=0;
    for (int nv : {0,1,3,510,511,512,1514,2047,8192})
    for (int pattern=0;pattern<5;++pattern)
    for (int tail=0;tail<4;++tail) {
        const int requested=511,kp=4,nsel=requested*kp+kp,pos=nv*kp+tail;
        const int keep=std::min(nv,requested);
        std::vector<float> scores(std::max(nv,1));
        for(int j=0;j<nv;++j) {
            if(pattern==0)scores[j]=0.f;                         // all tied
            if(pattern==1)scores[j]=(j<510?2.f:(j<512?1.f:-1.f)); // tie straddles cutoff
            if(pattern==2)scores[j]=float((j*31)%17-8);          // signed repeated levels
            if(pattern==3)scores[j]=-float(j+1);                // unique negative
            if(pattern==4)scores[j]=float((j*131)%8191)/8192.f;  // shuffled, finely spaced
        }
        std::vector<int> ranked(nv);std::iota(ranked.begin(),ranked.end(),0);
        std::sort(ranked.begin(),ranked.end(),[&](int a,int b){
            return scores[a]!=scores[b]?scores[a]>scores[b]:a<b;
        });
        ranked.resize(keep);std::sort(ranked.begin(),ranked.end());
        std::vector<int> expected(nsel,-1),got(nsel);
        for(int i=0;i<keep;++i)for(int k=0;k<kp;++k)expected[i*kp+k]=ranked[i]*kp+k;
        for(int k=0;k<=tail;++k)expected[requested*kp+k]=nv*kp+k;
        float* ds=nullptr;int *out=nullptr,*dp=nullptr;
        cudaMalloc(&ds,scores.size()*4);cudaMalloc(&out,nsel*4);cudaMalloc(&dp,4);
        cudaMemcpy(ds,scores.data(),scores.size()*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dp,&pos,4,cudaMemcpyHostToDevice);
        cudaMemset(out,0x7f,nsel*4); // expose any unwritten slot
        select_k<<<1,kBlock>>>(ds,out,nv,nsel,kp,requested,dp,true);
        auto rc=cudaMemcpy(got.data(),out,nsel*4,cudaMemcpyDeviceToHost);
        if(rc!=cudaSuccess||got!=expected) {
            std::fprintf(stderr,"FAIL nv=%d pattern=%d tail=%d cuda=%s\n",nv,pattern,tail,cudaGetErrorString(rc));
            return 1;
        }
        cudaFree(ds);cudaFree(out);cudaFree(dp);++cases;
    }
    std::printf("PASS %d cases: CUDA selection equals CPU sorted top-k, every slot initialized\n",cases);
}
"""
with tempfile.TemporaryDirectory(prefix="hy4-flash-indexer-ties-") as work:
    work = Path(work)
    cu, binary = work / "test.cu", work / "test"
    cu.write_text(header + kernel + main)
    subprocess.run([os.environ.get("NVCC", "nvcc"), "-O2", str(cu), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
