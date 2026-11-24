// Weight-stationary Q4_K GU for a small speculative window. Each group is the
// first occurrence of a slot; all its token entries share the decoded weight.
template<int C, int R>
__global__ void hy4_flash_window_q4_pair_k(const Q4KBlock* gate,const Q4KBlock* up,
    std::size_t gs,std::size_t us,const Q81Block* x,std::size_t xs,float* yg,float* yu,
    int ni,int no,int entries,const int32_t* index,int half_grid) {
    const int first=blockIdx.y,slot=index[first];
    if(slot<0)return;
    for(int i=0;i<first;++i)if(index[i]==slot)return;
    int members[C],nc=0;
    for(int i=first;i<entries;++i)if(index[i]==slot && nc<C)members[nc++]=i;
    const bool upper=blockIdx.x>=half_grid;
    const int row0=(blockIdx.x-(upper?half_grid:0))*R;
    const auto* w=(upper?up:gate)+slot*(upper?us:gs);
    float* y=upper?yu:yg;
    const int tid=threadIdx.y*WARP+threadIdx.x;
    const int iqs=VDR*(tid%(QI/VDR));
    const int off=2*((iqs/2)/4),j=off/2,jm=j&1;
    const uint32_t hi=uint32_t(-int32_t(j>=2));
    const int nb=ni/QK;
    float acc[C][R]={};
    for(int kb=tid/(QI/VDR);kb<nb;kb+=VDR*WARPS*WARP/QI) {
        int u[C][4];float d8[C][2];
        #pragma unroll
        for(int c=0;c<C;++c)if(c<nc) {
            const auto* q=x+members[c]*xs+kb*(QK/Q8K)+off;
            #pragma unroll
            for(int z=0;z<2;++z) {
                d8[c][z]=__low2float(q[z].ds);
                const int* a=reinterpret_cast<const int*>(q[z].qs)+((iqs/2)%4);
                u[c][2*z]=a[0];u[c][2*z+1]=a[4];
            }
        }
        #pragma unroll
        for(int r=0;r<R;++r)if(row0+r<no) {
            const auto* b=w+(row0+r)*nb+kb;
            const int* q=reinterpret_cast<const int*>(b->qs+16*off+4*((iqs/2)%4));
            const int v[2]={q[0],q[4]};
            const auto* sc=reinterpret_cast<const uint16_t*>(b->scales);
            const uint32_t s0=sc[jm],s2=sc[jm+2],s4=sc[jm+4];
            uint16_t aux[2];
            aux[0]=uint16_t(((s0&0x3f3f)&~hi)|((((s4>>0)&0x0f0f)|((s0&0xc0c0)>>2))&hi));
            aux[1]=uint16_t(((s2&0x3f3f)&~hi)|((((s4>>4)&0x0f0f)|((s2&0xc0c0)>>2))&hi));
            const auto* scales=reinterpret_cast<const uint8_t*>(aux);
            const half2 dm=b->dm;
            #pragma unroll
            for(int c=0;c<C;++c)if(c<nc)acc[c][r]+=q4_q8_dot_impl(v,u[c],scales,scales+2,dm,d8[c]);
        }
    }
    __shared__ float partial[WARPS-1][C][R][WARP];
    if(threadIdx.y>0) {
        #pragma unroll
        for(int c=0;c<C;++c)
            #pragma unroll
            for(int r=0;r<R;++r)partial[threadIdx.y-1][c][r][threadIdx.x]=acc[c][r];
    }
    __syncthreads();
    if(threadIdx.y>0)return;
    #pragma unroll
    for(int c=0;c<C;++c)if(c<nc) {
        #pragma unroll
        for(int r=0;r<R;++r) {
            #pragma unroll
            for(int z=0;z<WARPS-1;++z)acc[c][r]+=partial[z][c][r][threadIdx.x];
            acc[c][r]=warp_sum(acc[c][r]);
            if(threadIdx.x==r && row0+r<no)y[members[c]*no+row0+r]=acc[c][r];
        }
    }
}
