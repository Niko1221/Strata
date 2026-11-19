#include "strata/artifact/gguf_reader.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
template<class T>void put(std::ofstream& f,T x){f.write(reinterpret_cast<const char*>(&x),sizeof(x));}
void str(std::ofstream& f,const std::string& x){put<uint64_t>(f,x.size());f.write(x.data(),x.size());}
int main(){auto p=std::filesystem::temp_directory_path()/("q38-metadata-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".gguf");try{
    auto make=[&](int count){std::ofstream f(p,std::ios::binary);put<uint32_t>(f,0x46554747);put<uint32_t>(f,3);put<uint64_t>(f,0);put<uint64_t>(f,1);str(f,"selected");put<uint32_t>(f,9);put<uint32_t>(f,5);put<uint64_t>(f,count);for(int i=0;i<count;++i)put<int32_t>(f,i%2?1:-1);};
    make(28672);{strata::GgufFile old(p.string());if(old.get("selected")->items.size()!=64)throw std::runtime_error("default behavior changed");}
    {strata::GgufFile full(p.string(),{"selected"});auto v=full.get("selected");if(v->items.size()!=28672||int64_t(v->items[28670].u)!=-1)throw std::runtime_error("retained array wrong");}
    make(65537);bool refused=false;try{strata::GgufFile over(p.string(),{"selected"});}catch(const std::exception&){refused=true;}if(!refused)throw std::runtime_error("retention cap missing");
    make(248320);{strata::GgufFile vocabulary(p.string(),{"selected"},true,248320);if(vocabulary.get("selected")->items.size()!=248320)throw std::runtime_error("complete vocabulary not retained");}
    // The opt-in vocabulary count must not permit independently unbounded strings.
    {std::ofstream f(p,std::ios::binary);put<uint32_t>(f,0x46554747);put<uint32_t>(f,3);put<uint64_t>(f,0);put<uint64_t>(f,1);str(f,"selected");put<uint32_t>(f,9);put<uint32_t>(f,8);put<uint64_t>(f,1);put<uint64_t>(f,65ull*1024*1024);f.seekp(66ull*1024*1024);f.put(0);}
    refused=false;try{strata::GgufFile huge(p.string(),{"selected"},true,248320);}catch(const std::exception&){refused=true;}if(!refused)throw std::runtime_error("aggregate vocabulary header cap missing");
    {std::ofstream f(p,std::ios::binary);put<uint32_t>(f,0x46554747);put<uint32_t>(f,3);put<uint64_t>(f,0);put<uint64_t>(f,1);str(f,"selected");put<uint32_t>(f,9);put<uint32_t>(f,8);put<uint64_t>(f,20);for(int i=0;i<20;++i)str(f,std::string(32,'a'));}
    refused=false;try{strata::GgufFile aggregate(p.string(),{"selected"},true,248320,256);}catch(const std::exception&){refused=true;}if(!refused)throw std::runtime_error("aggregate small strings bypass header cap");
    for(bool array : {false,true})for(bool reverse : {false,true}) {
        const std::string key=array?"prism.hadamard.sign_values":"prism.hadamard.version";
        {std::ofstream f(p,std::ios::binary);put<uint32_t>(f,0x46554747);put<uint32_t>(f,3);put<uint64_t>(f,0);put<uint64_t>(f,2);
            for(int i=0;i<2;++i){str(f,key);put<uint32_t>(f,array?9:4);if(array){put<uint32_t>(f,5);put<uint64_t>(f,1);}put<int32_t>(f,(i==int(reverse))?1:2);}}
        {strata::GgufFile legacy(p.string());const auto* v=legacy.get(key);if((array?v->items[0].u:v->u)!=uint64_t(reverse?2:1))throw std::runtime_error("legacy duplicate behavior changed");}
        refused=false;try{strata::GgufFile strict(p.string(),{key},true);}catch(const std::exception& e){refused=std::string(e.what()).find("duplicate metadata key")!=std::string::npos;}
        if(!refused)throw std::runtime_error("duplicate metadata accepted in strict mode");
    }
    std::filesystem::remove(p);std::cout<<"PASS default sampling, complete selected array/vocabulary, count and aggregate byte caps, strict duplicate scalars/arrays in both orders\n";return 0;
}catch(const std::exception& e){std::filesystem::remove(p);std::cerr<<e.what()<<"\n";return 1;}}
