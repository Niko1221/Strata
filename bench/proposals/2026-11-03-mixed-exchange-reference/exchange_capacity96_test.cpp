#include "strata/core/exchange_storage.hpp"
#include <vector>
#include <cstring>
#include <iostream>
#include <stdexcept>
using strata::core::detail::ExchangeStorage;
void need(bool x){if(!x)throw std::runtime_error("96-slot ownership/capacity failure");}
int main(){constexpr size_t count=96;std::vector<size_t> types={7,31,65},sizes(6*count);std::vector<uint64_t> off(6*count,~uint64_t{0});size_t total=0;
 for(size_t t=0;t<3;++t)for(size_t q=0;q<2*count;++q){size_t i=t*2*count+q;sizes[i]=types[t];if(q<count){off[i]=total;total+=types[t];}}
 std::vector<uint8_t> h(total),d(total),p(total),pd(total);ExchangeStorage s;std::string e;need(s.initialize_mixed(off,sizes,h.data(),d.data(),total,p.data(),pd.data(),count,e));
 need(!s.commit_for(0,3*count,0,s.spare_for(31,0).host,31));
 for(size_t pass=0;pass<2;++pass)for(size_t t=0;t<3;++t){std::vector<ExchangeStorage::View> before(count),stage(count);
  for(size_t q=0;q<count;++q){size_t in=t*2*count+q+(pass?count:0);before[q]=s.resident(in);stage[q]=s.spare_for(types[t],q);need(before[q].host&&stage[q].host);std::memset(stage[q].host,int(17+t+pass),types[t]);}
  for(size_t q=0;q<count;++q){size_t in=t*2*count+q+(pass?count:0),out=t*2*count+q+(pass?0:count);need(s.commit_for(in,out,q,stage[q].host,types[t]));auto v=s.resident(out);need(v.host==stage[q].host&&v.device==stage[q].device&&v.bytes==types[t]);need(s.spare_for(types[t],q).host==before[q].host);for(size_t j=0;j<types[t];++j)need(v.host[j]==17+t+pass);}
  need(!s.spare_for(types[t],count).host);
 }
 need(s.exchanges()==576);need(s.avoided_bytes()==total*2);s.clear();need(!s.active()&&!s.resident(0).host&&!s.spare_for(7,0).host);std::cout<<"three size buckets, 96 simultaneous exchanges each, forward/reverse, cross-size rejection and clear: PASS\n";
}
