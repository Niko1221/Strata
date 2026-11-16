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
    std::filesystem::remove(p);std::cout<<"PASS default sampling, complete selected array, retention cap\n";return 0;
}catch(const std::exception& e){std::filesystem::remove(p);std::cerr<<e.what()<<"\n";return 1;}}
