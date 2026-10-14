// WIP SM75 QK/PV FP16 MMA with FP32 accumulators. Included inside the kernel namespace.
namespace wm = nvcuda::wmma;
template<int MODE,int CH> __global__ void __launch_bounds__(256) qsa_padded_mma_kernel(const float* q,QsaAttnPools p,const int* ids,const int* step,int nkh,int ps,float scale,float* pa,float* pm,float* pl,int nc,int cap,long long stride){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750

 q+=blockIdx.z*nkh*12*256;ids+=blockIdx.z*cap;step+=blockIdx.z*kStepCount;pa+=blockIdx.z*stride;pm+=blockIdx.z*stride;pl+=blockIdx.z*stride;
 __shared__ __align__(32) __half kv[CH][264];
 __shared__ __align__(32) __half qh[16][264];
 __shared__ __align__(32) float sp[16][CH+8];

 __shared__ long long rows[CH];
 extern __shared__ __align__(32) float warp_tmp[];
 __half (*ph)[CH+8]=reinterpret_cast<__half (*)[CH+8]>(warp_tmp+2048);
 int t=threadIdx.x,lane=t&31,warp=t>>5,kvh=blockIdx.y,chunk=blockIdx.x,c0=chunk*CH;
 int nh=min(CH,__ldg(step+kStepWidth)-c0),slot=kvh*nc+chunk;
 if(nh<=0){if(t<12){pm[slot*12+t]=-FLT_MAX;pl[slot*12+t]=0;}return;}
 if(t<CH){long long r=-1;if(t<nh){int cell=ids[c0+t];int page=p.page_table[cell/ps];if(page>=0)r=((long long)page*nkh+kvh)*ps+cell%ps;}rows[t]=r;}
 // Convert the twelve query heads to FP16 and pad the MMA tile to sixteen.
 for(int i=t;i<16*256;i+=256)qh[i/256][i%256]=__float2half_rn(i/256<12?q[kvh*12*256+i]:0);
 __syncthreads();
 {
  for(int i=t*8;i<CH*256;i+=256*8){int c=i/256,d=i%256;float v[8];if(c<nh&&rows[c]>=0)load8<MODE>(p,false,rows[c],d,v);else for(int j=0;j<8;++j)v[j]=0;
#pragma unroll
   for(int j=0;j<8;++j)kv[c][d+j]=__float2half_rn(v[j]);
  }
  __syncthreads();
  if(warp<CH/16){wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> a;wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major>b;wm::fragment<wm::accumulator,16,16,16,float>acc;
   wm::fill_fragment(acc,0.f);
   for(int k=0;k<256;k+=16){wm::load_matrix_sync(a,&qh[0][k],264);wm::load_matrix_sync(b,&kv[warp*16][k],264);wm::mma_sync(acc,a,b,acc);}
   wm::store_matrix_sync(&sp[0][warp*16],acc,CH+8,wm::mem_row_major);
  }
  __syncthreads();
  for(int i=t;i<12*CH;i+=256){int h=i/CH,c=i%CH;sp[h][c]=(c<nh&&rows[c]>=0)?sp[h][c]*scale:-FLT_MAX;}
 }
 __syncthreads();
 for(int h=warp;h<12;h+=8){float m=-FLT_MAX;
#pragma unroll
  for(int c=lane;c<CH;c+=32)m=fmaxf(m,sp[h][c]);m=warp_max(m);float sum=0;
#pragma unroll
  for(int c=lane;c<CH;c+=32){float e=c<nh&&rows[c]>=0?__expf(sp[h][c]-m):0;sp[h][c]=e;sum+=e;}
  sum=warp_sum(sum);if(lane==0){pm[slot*12+h]=m;pl[slot*12+h]=sum;}
 }
 __syncthreads();
 {
  for(int i=t;i<16*CH;i+=256)ph[i/CH][i%CH]=__float2half_rn(i/CH<12?sp[i/CH][i%CH]:0);
  for(int i=t*8;i<CH*256;i+=256*8){int c=i/256,d=i%256;float v[8];if(c<nh&&rows[c]>=0)load8<MODE>(p,true,rows[c],d,v);else for(int j=0;j<8;++j)v[j]=0;
#pragma unroll
   for(int j=0;j<8;++j)kv[c][d+j]=__float2half_rn(v[j]);
  }
  __syncthreads();
  for(int col=warp*16;col<256;col+=8*16){wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major>a;wm::fragment<wm::matrix_b,16,16,16,__half,wm::row_major>b;wm::fragment<wm::accumulator,16,16,16,float>acc;wm::fill_fragment(acc,0.f);
   for(int k=0;k<CH;k+=16){wm::load_matrix_sync(a,&ph[0][k],CH+8);wm::load_matrix_sync(b,&kv[k][col],264);wm::mma_sync(acc,a,b,acc);}
   float* tmp=warp_tmp+warp*256;wm::store_matrix_sync(tmp,acc,16,wm::mem_row_major);__syncwarp();
   for(int i=lane;i<12*16;i+=32)pa[((size_t)slot*12+i/16)*256+col+i%16]=tmp[i];__syncwarp();
  }
 }
#endif
}
