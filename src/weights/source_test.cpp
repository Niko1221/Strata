#include "strata/weights/safetensors.hpp"
#include "json_checked.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

namespace w = strata::weights;
using w::detail::Json;
using w::detail::require;
namespace fs = std::filesystem;
int main() {
    // The directory is newly created by this test, never a user's model path.
    const auto root = fs::temp_directory_path() / ("strata-source-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        require(fs::create_directory(root), "test directory collision");
        const std::string expert = "model.language_model.layers.0.mlp.experts.0.weight";
        const std::string mtp = "mtp.layers.0.mlp.experts.0.weight";
        Json header;
        for (const auto& [name,offset] : std::vector<std::pair<std::string,int>>{{"a",0},{"b",4},{expert,8},{mtp,12}})
            header[name] = {{"dtype","U8"},{"shape",{4}},{"data_offsets",{offset,offset+4}}};
        const auto text = header.dump();
        {
            std::ofstream f(root/"test.safetensors",std::ios::binary);
            const uint64_t n=text.size();
            f.write(reinterpret_cast<const char*>(&n),8); f.write(text.data(),text.size());
            const uint8_t data[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
            f.write(reinterpret_cast<const char*>(data),sizeof data);
            std::ofstream index(root/"model.safetensors.index.json");
            index << Json{{"weight_map",{{"a","test.safetensors"},{"b","test.safetensors"},{expert,"test.safetensors"},{mtp,"test.safetensors"}}}};
        }
        {
            w::SafetensorsSource source(root);
            std::array<uint8_t,4> a{},b{},e{};
            const w::ReadRequest reads[] = {{&source.tensor("b"),0,b},{&source.tensor(expert),0,e},{&source.tensor("a"),0,a}};
            source.read_many(reads);
            require(a==std::array<uint8_t,4>{0,1,2,3} && b==std::array<uint8_t,4>{4,5,6,7},"scatter destination/order mismatch");
            require(source.io_stats().data_calls==2,"adjacent dense reads were not coalesced or family boundary ignored");
            require(source.io_stats().data_bytes[0]==8 && source.io_stats().data_bytes[1]==4,"family accounting mismatch");
            auto before=source.io_stats().data_calls;
            auto copied=source.tensor("a");
            const w::ReadRequest foreign{&copied,0,a};
            bool refused=false;
            try { source.read_many({&foreign,1}); } catch(const std::runtime_error&) { refused=true; }
            require(refused && source.io_stats().data_calls==before,"foreign descriptor accepted");
            const w::ReadRequest invalid[] = {{&source.tensor("a"),0,a},{&source.tensor("b"),3,b}};
            refused=false;
            try { source.read_many(invalid); } catch(const std::runtime_error&) { refused=true; }
            require(refused && source.io_stats().data_calls==before,"batch read before validating all ranges");
            source.seal_expert_reads();
            refused=false;
            try { source.read(source.tensor(expert)); } catch(const std::runtime_error&) { refused=true; }
            require(refused && source.io_stats().data_calls==before,"expert reads continued after sealing");
            refused=false;
            try { source.read(source.tensor(mtp)); } catch(const std::runtime_error&) { refused=true; }
            require(refused && source.io_stats().data_calls==before,"MTP reads continued after sealing");
            // Reuse the owned sample after sealing; storage addresses stay fixed.
            const auto* address=e.data();
            for (int i=0;i<100;++i) require(e.data()==address && e[3]==11,"resident sample changed");
            require(source.io_stats().data_calls==before,"sample reuse read the source");
            source.read(source.tensor("a")); // sealing experts must not disable other families
            before=source.io_stats().data_calls;
            source.seal_resident_reads();
            source.seal_resident_reads(); // idempotent; descriptions remain owned
            source.read_many({});
            refused=false;
            try { source.read(source.tensor("a")); } catch(const std::runtime_error&) { refused=true; }
            require(refused && source.io_stats().data_calls==before,"resident read reopened a startup file");
            require(source.tensor("a").bytes==4 && source.shards().size()==1,"sealing lost metadata");
        }
        const auto cleanup = fs::canonical(root);
        require(cleanup.parent_path()==fs::canonical(fs::temp_directory_path()) &&
                cleanup.filename()==root.filename(), "refusing cleanup outside test directory");
        fs::remove_all(cleanup);
        std::cout << "PASS: sorted coalesced reads, family counters, bounds, ownership, sealed expert source and sample reuse\n";
        return 0;
    } catch(const std::exception& e) {
        // Deliberately retain failed fixtures for diagnosis.
        std::cerr << e.what() << '\n'; return 1;
    }
}
