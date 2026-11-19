#include "strata/core/native_cache_manifest.hpp"
#include "strata/core/layout.hpp"
#include <filesystem>
#include <fstream>
#include <cstring>
#include <set>
#include <limits>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/stat.h>
#endif
extern "C" {
#include "../../spin8/third_party/sha256/sha256.h"
}
namespace strata::core {
namespace {
namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;
void require(bool v, const char *why) { if (!v) throw std::runtime_error(why); }
NativeCacheDigest hash(const Bytes &v) {
  NativeCacheDigest d{}; spin8_sidecar_sha256_hash(d.data(),v.data(),v.size()); return d;
}
std::string hex(const NativeCacheDigest &d) {
  std::string s; for(auto x:d) { s += "0123456789abcdef"[x>>4]; s += "0123456789abcdef"[x&15]; } return s;
}
Bytes read(const fs::path &p,size_t bound) {
  const auto n=fs::file_size(p); require(n<=bound,"native manifest: file exceeds bound");
  Bytes v(static_cast<size_t>(n)); std::ifstream f(p,std::ios::binary);
  require(bool(f) && bool(f.read(reinterpret_cast<char*>(v.data()),v.size())),"native manifest: read failed");
  require(f.peek()==std::char_traits<char>::eof(),"native manifest: file changed while reading"); return v;
}
uint64_t ticks(const fs::path &p) {
#ifdef _WIN32
  WIN32_FILE_ATTRIBUTE_DATA a{};
  require(GetFileAttributesExW(p.c_str(),GetFileExInfoStandard,&a)!=0,"native manifest: file metadata unavailable");
  return ((uint64_t(a.ftLastWriteTime.dwHighDateTime)<<32)|a.ftLastWriteTime.dwLowDateTime)-116444736000000000ull;
#else
  struct stat s{}; require(::stat(p.c_str(),&s)==0,"native manifest: stat failed");
  return uint64_t(s.st_mtim.tv_sec)*10000000+uint64_t(s.st_mtim.tv_nsec)/100;
#endif
}
struct Reader {
  const Bytes &v; size_t at=0;
  Bytes bytes(size_t n) { require(n<=v.size()-at,"native manifest: truncated capsule"); Bytes b(v.begin()+at,v.begin()+at+n);at+=n;return b; }
  uint64_t integer(size_t n=4) { auto b=bytes(n);uint64_t x=0;for(size_t i=0;i<n;++i)x|=uint64_t(b[i])<<(i*8);return x; }
  double real() { auto x=integer(8);double d;std::memcpy(&d,&x,8);return d; }
  std::string text() {
    auto n=integer();require(n>0 && n<=32768,"native manifest: invalid path size");auto b=bytes(size_t(n));
    // Strict UTF-8, no NUL, overlong, surrogate, or out-of-range encodings.
    for(size_t i=0;i<b.size();) { uint32_t c=b[i++]; if(c<128) {require(c!=0,"native manifest: NUL path");continue;}
      int count=c>=0xf0 && c<=0xf4?3:c>=0xe0 && c<=0xef?2:c>=0xc2 && c<=0xdf?1:0;require(count!=0 && i+count<=b.size(),"native manifest: invalid UTF-8");
      uint32_t value=c&((1u<<(6-count))-1);for(int j=0;j<count;++j){require((b[i]&0xc0)==0x80,"native manifest: invalid UTF-8");value=(value<<6)|(b[i++]&63);}
      require(value>=(count==1?128u:count==2?2048u:65536u) && value<=0x10ffff && !(value>=0xd800 && value<=0xdfff),"native manifest: invalid UTF-8");
    }
    return std::string(b.begin(),b.end());
  }
  NativeCacheDigest digest() { auto b=bytes(32);NativeCacheDigest d{};std::copy(b.begin(),b.end(),d.begin());return d; }
};
void word(Bytes &v,uint64_t x,size_t n=8) {for(size_t i=0;i<n;++i)v.push_back(uint8_t(x>>(i*8)));}
void real(Bytes &v,double x) {uint64_t b;std::memcpy(&b,&x,8);word(v,b);}
void metadata(Reader &r,const fs::path &p,bool with_hash) {
  const uint64_t size=r.integer(8), time=r.integer(8);auto digest=r.digest();
  require(fs::file_size(p)==size && ticks(p)==time,"native manifest: selected file metadata changed");
  if(with_hash) require(hash(read(p,32u*1024*1024))==digest,"native manifest: selected metadata content changed");
}
uint32_t dialect(const std::string &s) {
  if(s.size()==6 && s.substr(0,5)=="turbo" && s[5]>='2' && s[5]<='6')return s[5]-'1';
  throw std::runtime_error("native manifest: cache type must be turbo2 through turbo6");
}
}
bool native_manifest_load(const std::string &path,const std::string &expected,
                          const NativeManifestInputs &in,NativeSessionConfig &out,std::string &error) {
  try {
    require(out.plans.empty(),"native manifest: output already bound");
    auto bytes=read(fs::u8path(path),16u*1024*1024);
    require(expected.size()==64 && hex(hash(bytes))==expected,"native manifest: independently expected capsule SHA mismatch");
    Reader r{bytes};require(r.bytes(8)==Bytes({'S','T','N','C','A','P','1','\n'}),"native manifest: magic/version mismatch");
    require(r.integer()==1,"native manifest: unknown version");
    NativeCacheRequested req;req.k=NativeKvFormat(r.integer());req.v=NativeKvFormat(r.integer());req.representation=NativeCacheRepresentation(r.integer());req.view_mask=uint32_t(r.integer());
    for(auto &a:req.alpha)a=r.real();
    req.capacity=r.integer(8);req.max_queries=req.max_append_rows=r.integer(8);
    require(req.capacity==in.capacity && req.max_queries==in.rows,"native manifest: execution limits differ");
    if(!in.k.empty())require(uint32_t(req.k)==dialect(in.k),"native manifest: explicit K type conflicts");
    if(!in.v.empty())require(uint32_t(req.v)==dialect(in.v),"native manifest: explicit V type conflicts");
    const auto planes=r.integer();require(planes<=64,"native manifest: excessive planes");
    for(uint64_t i=0;i<planes;++i) {auto a=r.integer(),b=r.integer();double angle=r.real();req.planes.push_back({uint32_t(a),uint32_t(b),angle});}
    NativeCacheBinding binding;
    Bytes position;const char domain[]="strata-native-position/v1";position.insert(position.end(),domain,domain+sizeof(domain));
    word(position,0);word(position,uint64_t(in.geometry.n_rot));word(position,uint64_t(in.rope.type));
    for(auto x:{in.rope.freq_base,in.rope.factor,in.rope.freq_scale_in,in.rope.orig_ctx,in.rope.ext_factor,in.rope.attn_factor,in.rope.beta_fast,in.rope.beta_slow})real(position,x);
    binding.position=hash(position);require(r.digest()==binding.position,"native manifest: resolved position differs");
    const auto count=r.integer();require(count==in.sources.size() && count>0 && count<=256,"native manifest: source count differs");
    for(size_t i=0;i<count;++i) {
      const auto selected=fs::u8path(in.sources[i]);require(fs::equivalent(fs::u8path(r.text()),selected),"native manifest: selected source path differs");
      auto d=r.digest();require(d!=NativeCacheDigest{},"native manifest: missing raw source digest");binding.raw_models.push_back(d);metadata(r,selected,false);
    }
    require(fs::equivalent(fs::u8path(r.text()),fs::u8path(in.pack)),"native manifest: selected pack differs");
    const auto files=r.integer();require(files>1 && files<=256,"native manifest: invalid pack file directory");std::set<std::string> seen;
    for(size_t i=0;i<files;++i) {
      auto name=r.text();auto p=fs::u8path(name);require(!p.is_absolute() && !p.has_root_path() && p.lexically_normal()==p,"native manifest: invalid relative pack path");
      for(const auto &part:p)require(part!="..","native manifest: escaping pack path");
      require(seen.insert(name).second,"native manifest: duplicate pack file");metadata(r,fs::u8path(in.pack)/p,name!="dense.bin");
    }
    require(seen.count("dense.bin") && seen.count("index.txt") && seen.count("native_experts.txt"),"native manifest: incomplete pack binding");
    const auto layers=r.integer();require(layers==uint64_t(in.geometry.n_qsa_layers()),"native manifest: normal layer count differs");
    for(int64_t l=0;l<in.geometry.n_layers;++l)if(is_qsa_layer(in.geometry,l)) {
      spin8_query_adapter_layer_spec spec{uint32_t(l),uint32_t(in.geometry.n_head),uint32_t(in.geometry.n_head_kv),uint32_t(in.geometry.head_dim)};
      require(r.integer()==spec.layer_index && r.integer()==spec.q_heads && r.integer()==spec.kv_heads && r.integer()==spec.head_width,"native manifest: actual logical layer geometry differs");binding.normal_layers.push_back(spec);
    }
    const auto sidecar_size=r.integer(8);require(sidecar_size<=8u*1024*1024,"native manifest: excessive adapter");auto sidecar=r.bytes(size_t(sidecar_size));require(r.at==bytes.size(),"native manifest: trailing input");
    require(!in.explicit_off || (req.view_mask==1 && req.planes.empty() && sidecar.empty()),"native manifest: explicit off conflicts with Spin views/query adaptation");
    NativeSessionConfig result;result.rows=size_t(in.rows);result.selection_width=size_t(std::min<uint64_t>(in.capacity,strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells,qsa_shapes(in.geometry))));
    for(auto &s:binding.normal_layers) {binding.selected_layer=s.layer_index;auto plan=NativeCachePlan::bind(req,binding,sidecar.empty()?nullptr:sidecar.data(),sidecar.size(),error);if(!plan)return false;result.plans.push_back(std::move(plan));}
    out=std::move(result);return true;
  } catch(const std::exception &e) {error=e.what();return false;}
}
} // namespace strata::core
