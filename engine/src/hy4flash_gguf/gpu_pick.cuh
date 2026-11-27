// Shared hy4-flash device argmax: preserve lowest-ID ties and reject any nonfinite
// input. The vocabulary is split across 64 blocks; no host round trip.
struct PickPart {float value;int id,bad;};
__global__ void pick_parts_k(const float* logits,int n,PickPart* parts,int excluded) {
    __shared__ float sv[8];__shared__ int si[8],sb[8];
    float best=-INFINITY;int id=n,bad=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        const float v=logits[i];bad|=!isfinite(v);if(i==excluded)continue;
        if(v>best||(v==best&&i<id)){best=v;id=i;}
    }
    for(int d=16;d;d/=2){float v=__shfl_down_sync(0xffffffff,best,d);int j=__shfl_down_sync(0xffffffff,id,d);
        bad|=__shfl_down_sync(0xffffffff,bad,d);if(v>best||(v==best&&j<id)){best=v;id=j;}}
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(!lane){sv[warp]=best;si[warp]=id;sb[warp]=bad;}__syncthreads();
    if(!warp){best=lane<8?sv[lane]:-INFINITY;id=lane<8?si[lane]:n;bad=lane<8?sb[lane]:0;
        for(int d=16;d;d/=2){float v=__shfl_down_sync(0xffffffff,best,d);int j=__shfl_down_sync(0xffffffff,id,d);
            bad|=__shfl_down_sync(0xffffffff,bad,d);if(v>best||(v==best&&j<id)){best=v;id=j;}}
        if(!lane)parts[blockIdx.x]={best,id,bad};}
}
__global__ void pick_merge_k(const PickPart* parts,int* token,int* output) {
    const int lane=threadIdx.x;float best=-INFINITY;int id=INT_MAX,bad=0;
    for(int i=lane;i<64;i+=32){auto p=parts[i];bad|=p.bad;if(p.value>best||(p.value==best&&p.id<id)){best=p.value;id=p.id;}}
    for(int d=16;d;d/=2){float v=__shfl_down_sync(0xffffffff,best,d);int j=__shfl_down_sync(0xffffffff,id,d);
        bad|=__shfl_down_sync(0xffffffff,bad,d);if(v>best||(v==best&&j<id)){best=v;id=j;}}
    if(!lane){*output=bad?-1:id;if(token)*token=*output;}
}
inline void device_pick(const float* logits,int n,PickPart* scratch,int* token,int* out,cudaStream_t stream,int excluded=-1) {
    pick_parts_k<<<64,256,0,stream>>>(logits,n,scratch,excluded);pick_merge_k<<<1,32,0,stream>>>(scratch,token,out);
}
