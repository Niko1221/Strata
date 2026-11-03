#include "strata/core/exchange_storage.hpp"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
using strata::core::detail::ExchangeStorage;
static constexpr uint64_t absent=~uint64_t{0};
void require(bool b,const char* msg){if(!b)throw std::runtime_error(msg);}
void exercise(size_t a,size_t b,size_t rounds){
 constexpr size_t n=24,keep=16,count=4,guard=31;
 std::vector<size_t> sizes(n);for(size_t i=0;i<n;++i)sizes[i]=i%2?a:b;
 std::vector<uint64_t> offsets(n,absent);size_t used=0;
 for(size_t i=0;i<keep;++i){offsets[i]=used;used+=sizes[i];}
 std::vector<uint8_t> ram(used+2*guard,0xcd),aliases(used),pool(count*(a+b)+2*guard,0xef),pa(count*(a+b));
 std::vector<std::vector<uint8_t>> reference(n);std::mt19937 rng(20261102);
 for(size_t i=0;i<n;++i){reference[i].resize(sizes[i]);for(auto& x:reference[i])x=uint8_t(rng());if(i<keep)std::memcpy(ram.data()+guard+offsets[i],reference[i].data(),sizes[i]);}
 ExchangeStorage storage;std::string err;require(storage.initialize_mixed(offsets,sizes,ram.data()+guard,aliases.data(),used,pool.data()+guard,pa.data(),count,err),err.c_str());
 std::vector<ExchangeStorage::View> physical;
 for(size_t i=0;i<keep;++i)physical.push_back(storage.resident(i));
 for(size_t bytes:{a,b})for(size_t q=0;q<count;++q)physical.push_back(storage.spare_for(bytes,q));
 std::set<uint8_t*> original;for(auto v:physical){require(v.host&&v.device&&v.bytes,"missing view");require(original.insert(v.host).second,"duplicate initial ownership");}
 std::atomic<bool> stop{false},failed{false};
 std::thread reader([&]{while(!stop.load()){for(size_t i=0;i<n;++i){auto v=storage.resident(i);if(v.host&&std::none_of(physical.begin(),physical.end(),[&](auto p){return p.host==v.host&&p.device==v.device&&p.bytes==v.bytes;}))failed=true;}}});
 struct Guard {std::atomic<bool>& s;std::thread& t;~Guard(){s=true;t.join();}} join{stop,reader};
 uint64_t swaps=0,avoided=0;
 for(size_t round=0;round<rounds;++round){
  std::vector<size_t> ins,outs;for(size_t i=0;i<n;++i)(storage.resident(i).host?ins:outs).push_back(i);
  std::shuffle(ins.begin(),ins.end(),rng);std::shuffle(outs.begin(),outs.end(),rng);
  std::vector<size_t> selected;
  for(size_t q=0;q<count;++q){
   size_t in=ins[q];auto it=std::find_if(outs.begin(),outs.end(),[&](auto out){return sizes[out]==sizes[in]&&std::find(selected.begin(),selected.end(),out)==selected.end();});
   if(it==outs.end())continue;size_t out=*it;selected.push_back(out);const size_t bytes=sizes[in];auto before=storage.resident(in);auto spare=storage.spare_for(bytes,q);
   require(spare.bytes==bytes,"bucket capacity mismatch");std::memcpy(spare.host,reference[out].data(),bytes);
   require(!std::memcmp(before.host,reference[in].data(),bytes),"input overwritten before commit");
   require(!storage.commit_for(in,out,q,spare.host,bytes+1),"wrong size accepted");
   require(!storage.commit_for(in,out,q,spare.host+1,bytes),"wrong staged pointer accepted");
   require(!storage.commit_for(in,in,q,spare.host,bytes),"self accepted");
   require(!storage.commit_for(in,out,count,spare.host,bytes),"bad q accepted");
   require(!storage.commit_for(n,out,q,spare.host,bytes),"bad expert accepted");
   require(storage.commit_for(in,out,q,spare.host,bytes),"valid mixed swap rejected");
   auto resident=storage.resident(out);auto recycled=storage.spare_for(bytes,q);
   require(resident.host==spare.host&&resident.device==spare.device,"alias not adopted");
   require(recycled.host==before.host&&recycled.device==before.device,"old buffer not recycled");
   require(!std::memcmp(recycled.host,reference[in].data(),bytes),"commit modified bytes");
   require(!storage.commit_for(in,out,q,spare.host,bytes),"duplicate commit accepted");++swaps;avoided+=bytes;
  }
  std::set<uint8_t*> live;
  for(size_t i=0;i<n;++i){auto v=storage.resident(i);if(v.host){require(v.bytes==sizes[i],"cross-size ownership");require(live.insert(v.host).second,"resident alias");require(!std::memcmp(v.host,reference[i].data(),sizes[i]),"resident corruption");}}
  for(size_t bytes:{a,b})for(size_t q=0;q<count;++q)require(live.insert(storage.spare_for(bytes,q).host).second,"spare aliases resident");
  require(live==original,"physical ownership lost");
 }
 require(!failed.load(),"concurrent lookahead saw invalid alias");require(storage.exchanges()==swaps&&storage.avoided_bytes()==avoided,"bad counters");
 require(std::all_of(ram.begin(),ram.begin()+guard,[](auto x){return x==0xcd;})&&std::all_of(ram.end()-guard,ram.end(),[](auto x){return x==0xcd;}),"RAM guard overwritten");
 require(std::all_of(pool.begin(),pool.begin()+guard,[](auto x){return x==0xef;})&&std::all_of(pool.end()-guard,pool.end(),[](auto x){return x==0xef;}),"pool guard overwritten");
 require(!storage.initialize_mixed(offsets,sizes,ram.data()+guard,aliases.data(),used,pool.data()+guard,pa.data(),count,err),"reinit live accepted");
 std::cout<<"mixed sizes "<<a<<","<<b<<" rounds "<<rounds<<" swaps "<<swaps<<" avoided "<<avoided<<" PASS\n";
}
void invalid(){uint8_t h[128]{},d[128]{},x[128]{},xd[128]{};ExchangeStorage s;std::string e;
 require(!s.initialize_mixed({0,4},{8,8},h,d,128,x,xd,2,e),"overlap accepted");
 require(!s.initialize_mixed({0,124},{8,8},h,d,128,x,xd,2,e),"bounds accepted");
 require(!s.initialize_mixed({0},{8,16},h,d,128,x,xd,2,e),"length mismatch accepted");
 require(!s.initialize_mixed({0},{0},h,d,128,x,xd,2,e),"zero accepted");
 require(!s.initialize_mixed({0},{8},h,nullptr,128,x,xd,2,e),"unmapped accepted");
 require(!s.initialize_mixed({0},{SIZE_MAX},h,d,128,x,xd,2,e),"overflow accepted");require(!s.active(),"failed init active");
 require(!s.spare_for(8,0).host,"inactive has spare");s.clear();require(!s.resident(0).host,"clear has expert");
}
int main(){try{invalid();exercise(33,97,4096);exercise(1510400,1305600,16);std::cout<<"bucket ownership/bytes/CUDA-alias/guards/concurrent lookahead: PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
