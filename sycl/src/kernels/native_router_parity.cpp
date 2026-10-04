#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_router.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
int main() {try {
 auto &q=dpct::get_in_order_queue();int calls=0;
 for(int ne:{512,256}) for(int nt:{1,6}) {
 auto x=sycl::malloc_device<float>(ne*nt,q);auto ids=sycl::malloc_device<int>(10*nt,q);auto w=sycl::malloc_device<float>(10*nt,q);
 std::vector<float> h(ne*nt),v(10*nt);std::vector<int> ix(10*nt);
 for(int rep=0;rep<5;++rep){
 for(int t=0;t<nt;++t)for(int i=0;i<ne;++i)h[t*ne+i]=rep==0?0.f:float(((i*73+t*19+rep*37)%ne)-ne/2)*.03125f;
 q.memcpy(x,h.data(),h.size()*4).wait();q.memset(ids,0xff,ix.size()*4).wait();q.memset(w,0xff,v.size()*4).wait();
 std::printf("begin ne=%d nt=%d rep=%d\n",ne,nt,rep);std::fflush(stdout);
 strata::kernels::native_router_top10_multi_ne(x,ids,w,nt,ne,&q);q.wait_and_throw();
 q.memcpy(ix.data(),ids,ix.size()*4).wait();q.memcpy(v.data(),w,v.size()*4).wait();
 for(int t=0;t<nt;++t){std::vector<int> ref(ne);std::iota(ref.begin(),ref.end(),0);
 std::stable_sort(ref.begin(),ref.end(),[&](int a,int b){return h[t*ne+a]>h[t*ne+b];});
 double sum=0,mx=h[t*ne+ref[0]];for(int k=0;k<10;++k)sum+=std::exp(double(h[t*ne+ref[k]])-mx);
 for(int k=0;k<10;++k){double want=std::exp(double(h[t*ne+ref[k]])-mx)/sum;
 if(ix[t*10+k]!=ref[k] || !std::isfinite(v[t*10+k]) || std::abs(v[t*10+k]-want)>2e-6){std::printf("FAIL ne=%d nt=%d token=%d k=%d id=%d expected=%d weight=%g expected=%g\n",ne,nt,t,k,ix[t*10+k],ref[k],v[t*10+k],want);return 1;}}
 }++calls;
 }sycl::free(x,q);sycl::free(ids,q);sycl::free(w,q);
 }std::printf("PASS %d changing-input calls; exact stable IDs and independent double softmax top10 reference\n",calls);return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
