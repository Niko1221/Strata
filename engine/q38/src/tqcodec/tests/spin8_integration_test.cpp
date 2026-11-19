#include "spin8/query.h"
#include "tqcodec/cache.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
  constexpr size_t d=9, rows=5;
  const spin8_plane planes[]={{0,1,.37},{2,3,-.51},{0,7,.23}};
  spin8_plan *spin=nullptr;
  if(spin8_plan_create(planes,3,&spin)!=SPIN8_OK) return 1;
  double U[3][d*d]{},H[64];
  if(spin8_hadamard_matrix(37,0,H)!=SPIN8_OK) return 2;
  for(int r=0;r<3;++r){double B[64];spin8_plan_matrix(spin,static_cast<spin8_view>(r),B);
    for(size_t i=0;i<8;++i)for(size_t j=0;j<8;++j)for(size_t k=0;k<8;++k)U[r][i*d+j]+=B[i*8+k]*H[k*8+j];
    U[r][8*d+8]=1.;}
  double kb[8],vb[16];for(size_t i=0;i<8;++i)kb[i]=-1+2.*i/7;for(size_t i=0;i<16;++i)vb[i]=-1+2.*i/15;
  tq_cache_config config{};config.width=d;config.capacity_rows=rows;config.key_bits=3;config.value_bits=4;
  for(int r=0;r<3;++r){config.transforms[r]=U[r];config.alpha[r]=1./3;}
  config.key_centroids=kb;config.value_centroids=vb;tq_cache *cache=nullptr;
  if(tq_cache_create(&config,&cache)!=TQ_OK)return 3;
  for(size_t n=0;n<rows;++n){double k[d],v[d];for(size_t j=0;j<d;++j){k[j]=std::sin(double(n*d+j));v[j]=std::cos(double(n*d+j));}
    if(tq_cache_append(cache,k,d,v,d)!=TQ_OK)return 4;}
  double down[8*d]{},up[56]{};for(size_t i=0;i<8*d;++i)down[i]=std::sin(double(i))*.1;up[0]=.25;up[13]=-.1;
  spin8_query_controller *ctl=nullptr;if(spin8_query_controller_create(d,down,sizeof(down),up,sizeof(up),&ctl)!=SPIN8_OK)return 5;
  double q[d],qp[d],actual[d],expected[d]{};for(size_t i=0;i<d;++i)q[i]=std::cos(double(i)*.3);
  if(spin8_query_controller_apply(ctl,SPIN8_F64,q,sizeof(q),d,qp,sizeof(qp),d,1)!=SPIN8_OK)return 6;
  if(qp[8]!=q[8])return 7;
  const double scale=1/std::sqrt(double(d));
  if(tq_cache_attend(cache,qp,d,nullptr,0,scale,actual,d)!=TQ_OK)return 8;
  // Independent dense expansion: restore each K to common coordinates, combine
  // scores, apply softmax once, then multiply the single shared V reconstruction.
  double scores[rows]{},vals[rows][d],maxscore=-1e300,sum=0;
  for(size_t n=0;n<rows;++n){uint8_t p[16];double norm,rotated[d];
    for(int r=0;r<3;++r){if(tq_cache_copy_row(cache,n,r,p,sizeof(p),&norm)!=TQ_OK)return 9;
      if(tq_decode_f64(p,sizeof(p),d,kb,3,norm,rotated)!=TQ_OK)return 10;
      for(size_t j=0;j<d;++j){double common=0;for(size_t i=0;i<d;++i)common+=U[r][i*d+j]*rotated[i];scores[n]+=qp[j]*common/3;}}
    scores[n]*=scale;maxscore=std::max(maxscore,scores[n]);
    if(tq_cache_copy_row(cache,n,3,p,sizeof(p),&norm)!=TQ_OK)return 11;
    if(tq_decode_f64(p,sizeof(p),d,vb,4,norm,vals[n])!=TQ_OK)return 12;}
  for(size_t n=0;n<rows;++n){double weight=std::exp(scores[n]-maxscore);sum+=weight;for(size_t j=0;j<d;++j)expected[j]+=weight*vals[n][j];}
  double maxerror=0;for(size_t j=0;j<d;++j)maxerror=std::max(maxerror,std::abs(expected[j]/sum-actual[j]));
  spin8_query_controller_free(ctl);spin8_plan_free(spin);tq_cache_destroy(cache);
  std::printf("strict Spin8 + HD + query controller + 3K/shared V max abs error %.17g\n",maxerror);
  return maxerror<1e-12?0:13;
}
