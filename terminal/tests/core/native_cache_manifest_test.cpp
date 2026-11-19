#include "native_cache_fixture.hpp"
#include "strata/core/native_cache_manifest.hpp"
#include <filesystem>
#include <fstream>
#include <cstdio>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
using namespace strata::core;
namespace fs=std::filesystem;
#define CHECK(x) do { if(!(x)) throw std::runtime_error(#x); } while(0)
using Bytes=std::vector<uint8_t>;
void word(Bytes &b,uint64_t x,size_t n=4) {for(size_t i=0;i<n;++i)b.push_back(uint8_t(x>>(i*8)));}
void real(Bytes &b,double x) {uint64_t z;std::memcpy(&z,&x,8);word(b,z,8);}
void text(Bytes &b,const std::string &s) {word(b,s.size());b.insert(b.end(),s.begin(),s.end());}
void digest(Bytes &b,const NativeCacheDigest &d) {b.insert(b.end(),d.begin(),d.end());}
void save(const fs::path &p,const Bytes &v) {std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size());CHECK(f.good());}
uint64_t ticks(const fs::path &p) {
  WIN32_FILE_ATTRIBUTE_DATA a{};CHECK(GetFileAttributesExW(p.c_str(),GetFileExInfoStandard,&a));
  return ((uint64_t(a.ftLastWriteTime.dwHighDateTime)<<32)|a.ftLastWriteTime.dwLowDateTime)-116444736000000000ull;
}
void metadata(Bytes &v,const fs::path &p) {
  word(v,3,8);word(v,ticks(p),8);digest(v,native_fixture::hash("abc",3));
}
int main() {
  try {
    auto root=fs::temp_directory_path()/fs::u8path("strata-native-manifest-"+std::to_string(GetCurrentProcessId()));
    CHECK(!fs::exists(root));fs::create_directory(root);
    auto source=root/"source.gguf",pack=root/"pack",cap=root/"config.bin";fs::create_directory(pack);
    save(source,{'a','b','c'});for(auto name:{"dense.bin","index.txt","native_experts.txt"})save(pack/name,{'a','b','c'});
    NativeManifestInputs in;in.sources={source.u8string()};in.pack=pack.u8string();in.capacity=8;in.rows=3;
    in.geometry.n_layers=8; // actual logical QSA layers 3 and 7
    Bytes v={'S','T','N','C','A','P','1','\n'};word(v,1);word(v,3);word(v,1);word(v,0);word(v,7);
    for(int i=0;i<3;++i)real(v,1./3);word(v,8,8);word(v,3,8);word(v,1);word(v,0);word(v,1);real(v,.37);
    Bytes position;const char domain[]="strata-native-position/v1";position.insert(position.end(),domain,domain+sizeof(domain));
    word(position,0,8);word(position,64,8);word(position,0,8);for(auto x:{1e7,1.,0.,262144.,0.,1.,32.,1.})real(position,x);
    digest(v,native_fixture::hash(position.data(),position.size()));word(v,1);
    const size_t source_path_at=v.size();
    text(v,source.u8string());digest(v,NativeCacheDigest{1});metadata(v,source);
    text(v,pack.u8string());word(v,3);for(auto name:{"dense.bin","index.txt","native_experts.txt"}){text(v,name);metadata(v,pack/name);}
    word(v,2);for(uint32_t l:{3,7}){word(v,l);word(v,24);word(v,2);word(v,256);}word(v,0,8);
    auto load=[&](const Bytes &bytes,NativeManifestInputs inputs) {
      save(cap,bytes);std::string error;NativeSessionConfig result;
      return native_manifest_load(cap.u8string(),native_fixture::hex(native_fixture::hash(bytes.data(),bytes.size())),inputs,result,error);
    };
    CHECK(load(v,in));
    // Reject malformed UTF-8 in the capsule parser itself, before a platform's
    // filesystem conversion can reject or reinterpret the path differently.
    for(uint8_t lead:{0x80,0xbf,0xc0,0xc1,0xf5,0xf6,0xf7,0xf8,0xfc,0xff}) {
      auto malformed=v;
      const size_t end=source_path_at+4+source.u8string().size();
      malformed.erase(malformed.begin()+source_path_at,malformed.begin()+end);
      const Bytes field={4,0,0,0,lead,0x90,0x80,0x80};
      malformed.insert(malformed.begin()+source_path_at,field.begin(),field.end());
      save(cap,malformed);std::string why;NativeSessionConfig rejected;
      CHECK(!native_manifest_load(cap.u8string(),native_fixture::hex(native_fixture::hash(malformed.data(),malformed.size())),in,rejected,why));
      if(why!="native manifest: invalid UTF-8") {
        std::fprintf(stderr,"lead 0x%02x reached platform path handling: %s\n",unsigned(lead),why.c_str());
        throw std::runtime_error("capsule parser must reject malformed UTF-8 before filesystem lookup");
      }
      CHECK(rejected.plans.empty());
    }
    // The maximum selected-cell width includes the current block tail.
    auto wider=v;for(size_t j=0;j<8;++j)wider[52+j]=uint8_t(uint64_t(4096)>>(8*j));
    auto wide_in=in;wide_in.capacity=4096;save(cap,wider);
    {std::string error;NativeSessionConfig cfg;CHECK(native_manifest_load(cap.u8string(),native_fixture::hex(native_fixture::hash(wider.data(),wider.size())),wide_in,cfg,error));CHECK(cfg.selection_width==2051);}

    auto changed=in;changed.explicit_off=true;CHECK(!load(v,changed));
    auto codec=v;codec[24]=1;
    const double simplex[3]={1.,0.,0.};std::memcpy(codec.data()+28,simplex,sizeof(simplex));
    codec[68]=0;codec.erase(codec.begin()+72,codec.begin()+88);
    CHECK(load(codec,changed));
    changed=in;changed.k="turbo5";CHECK(!load(v,changed));changed=in;changed.v="turbo3";CHECK(!load(v,changed));
    changed=in;changed.geometry.qsa_interval=2;CHECK(!load(v,changed));
    changed=in;changed.rope.freq_base=1e6;CHECK(!load(v,changed));
    changed=in;changed.capacity=9;CHECK(!load(v,changed));
    auto bad=v;bad.push_back(0);CHECK(!load(bad,in));bad=v;bad.resize(40);CHECK(!load(bad,in));
    bad=v;bad[8]=2;CHECK(!load(bad,in));
    save(cap,v);std::string error;NativeSessionConfig result;CHECK(!native_manifest_load(cap.u8string(),std::string(64,'0'),in,result,error));
    save(pack/"index.txt",{'d','e','f'});CHECK(!load(v,in));
    // All files are this test's fresh directory, never user input.
    fs::remove_all(root);
    std::puts("PASS native capsule: source/pack identity, position, actual layer directory, K/V/off conflicts, malformed input");return 0;
  } catch(const std::exception &e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
