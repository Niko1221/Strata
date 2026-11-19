#pragma once
#include <cstdint>
// Scalar packed weight math derived from ggml commit 643956891dfdf51aecdbd05ca508dd21e21e8195.
// Copyright (c) 2023-2026 The ggml authors, MIT; see LICENSE-formats.txt.
#ifdef __CUDACC__
#define Q38_IQ_HD __host__ __device__
#else
#define Q38_IQ_HD
#endif
namespace strata::qwen38 {
Q38_IQ_HD inline float iq_value(int type, const uint8_t* p, int i, float d, const uint64_t* g2, const uint32_t* g3) {
    const int sub=i/32, in=i%32, group=in/8, lane=in%8;
    if(type==22) {
        const int index=p[2+sub*4+group] | (((p[66+sub]>>(2*group))&3)<<8);
        const uint64_t grid=g2[index];
        const int magnitude=int((grid>>(8*lane))&255);
        const int scale=(p[74+sub]>>(4*(group/2)))&15;
        const float factor=d*(0.5f+scale)*0.25f;
        return factor*magnitude*((p[34+sub*4+group]&(1<<lane))?-1.f:1.f);
    }
    const int idx=in/4;
    const int index=p[2+sub*8+idx] | (((p[66+sub]>>idx)&1)<<8);
    const uint32_t grid=g3[index];
    const int magnitude=int((grid>>(8*(in%4)))&255);
    const int scale=(p[106+sub/2]>>(4*(sub%2)))&15;
    return (d*(1+2*scale))*magnitude*((p[74+sub*4+group]&(1<<lane))?-1.f:1.f);
}
}
#undef Q38_IQ_HD
