// Sequential ownership only: a microbenchmark, not an expert-cache benchmark.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
void check(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);} }
__global__ void sum_bytes(const unsigned char* p,size_t n,unsigned long long* result){
 __shared__ unsigned long long sums[256];
 unsigned long long s=0;
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)s+=p[i];
 sums[threadIdx.x]=s;__syncthreads();
 for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)sums[threadIdx.x]+=sums[threadIdx.x+stride];__syncthreads();}
 if(threadIdx.x==0)atomicAdd(result,sums[0]);
}
int main(){
 const size_t n=64ull<<20; unsigned long long* result;check(cudaMalloc(&result,sizeof(*result)));
 for(int mode=0;mode<3;++mode){
  unsigned char *host=nullptr,*gpu=nullptr;
  if(mode==2){check(cudaMallocManaged(&host,n));gpu=host;}
  else{check(cudaHostAlloc(&host,n,cudaHostAllocMapped));if(mode==0)check(cudaMalloc(&gpu,n));else check(cudaHostGetDevicePointer(&gpu,host,0));}
  std::vector<double> timings;
  for(int iter=0;iter<13;++iter){
   unsigned long long expected=0;
   for(size_t i=0;i<n;++i){host[i]=(unsigned char)((i+iter)%251);expected+=host[i];}
   auto start=std::chrono::steady_clock::now();
   check(cudaMemset(result,0,sizeof(*result)));
   if(mode==0)check(cudaMemcpy(gpu,host,n,cudaMemcpyHostToDevice));
   sum_bytes<<<256,256>>>(gpu,n,result);check(cudaGetLastError());check(cudaDeviceSynchronize());
   unsigned long long got;check(cudaMemcpy(&got,result,sizeof(got),cudaMemcpyDeviceToHost));
   if(got!=expected){std::fprintf(stderr,"checksum mismatch\n");return 1;}
   double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
   if(iter>=3)timings.push_back(ms);
  }
  std::sort(timings.begin(),timings.end());
  std::printf("%s bytes=%zu iterations=10 median_ms=%.6f min_ms=%.6f max_ms=%.6f checksum=PASS\n",mode==0?"pinned_copy":mode==1?"mapped_host":"managed_sequential",n,(timings[4]+timings[5])/2,timings.front(),timings.back());
  if(mode==2)check(cudaFree(host));else{check(cudaFreeHost(host));if(mode==0)check(cudaFree(gpu));}
 }
 check(cudaFree(result));
}
