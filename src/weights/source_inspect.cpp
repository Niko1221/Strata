#include "strata/weights/safetensors.hpp"
#include "json_checked.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>

namespace w = strata::weights;
using w::detail::Json;
using w::detail::require;
namespace {
uint64_t number(const std::string& text) {
    require(!text.empty() && text.find_first_not_of("0123456789") == text.npos,
            "expected unsigned decimal argument");
    return std::stoull(text);
}
int run(const std::vector<std::string>& args) {
    try {
        require(args.size() >= 3,
                "usage: strata-safetensors-source headers MODEL | read MODEL TENSOR OFFSET BYTES");
        const auto& mode = args[1];
        require(mode == "headers" || mode == "read", "unknown command");
        require(args.size() == (mode == "read" ? 6u : 3u), "wrong argument count");
        w::SafetensorsSource source(w::detail::utf8_path(args[2]));
        Json out{{"ok",true},{"tensor_count",source.tensors().size()},
                 {"scope","container validation only; not model inference"}};
        out["files"] = Json::array();
        for (const auto& shard : source.shards()) {
            const auto name = shard.path.filename().u8string();
            out["files"].push_back({{"name",std::string(name.begin(),name.end())},
                {"file_bytes",shard.file_bytes},{"header_bytes",shard.header_bytes},
                {"data_offset",shard.data_offset},{"tensor_count",shard.tensor_count}});
        }
        if (mode == "read") {
            const auto offset = number(args[4]), bytes = number(args[5]);
            require(bytes <= 4096, "diagnostic read limited to 4096 bytes");
            std::vector<uint8_t> buffer(static_cast<size_t>(bytes));
            const w::ReadRequest request{&source.tensor(args[3]),offset,buffer};
            source.read_many({&request,1});
            std::ostringstream hex;
            for (auto byte : buffer)
                hex << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(byte);
            out["hex"] = hex.str();
        }
        const auto& stats = source.io_stats();
        out["io"] = {{"header_bytes",stats.header_bytes},{"metadata_bytes",stats.metadata_bytes},
                     {"data_read_calls",stats.data_calls},{"staging_peak_bytes",stats.staging_peak_bytes}};
        for (size_t i=0;i<stats.data_bytes.size();++i)
            out["io"]["source_read_bytes"][w::family_name(static_cast<w::Family>(i))] = stats.data_bytes[i];
        std::cout << out.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << Json{{"ok",false},{"error",error.what()}}
            .dump(-1,' ',false,Json::error_handler_t::replace) << '\n';
        return 1;
    }
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i=0;i<argc;++i) {
        const auto text = std::filesystem::path(argv[i]).u8string();
        args.emplace_back(text.begin(),text.end());
    }
    return run(args);
}
#else
int main(int argc, char** argv) { return run({argv,argv+argc}); }
#endif
