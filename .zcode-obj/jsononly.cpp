// minimal: load pieces json, count, print
#include <nlohmann/json.hpp>
#include <cstdio>
#include <fstream>
int main(int argc, char** argv) {
    fprintf(stderr, "start\n");
    std::ifstream pf(argv[1]);
    fprintf(stderr, "opened ok=%d\n", (int) (bool) pf);
    nlohmann::json j;
    pf >> j;
    fprintf(stderr, "parsed size=%zu\n", j.size());
    fprintf(stderr, "first p=%s\n", j[0].at("p").get<std::string>().c_str());
    return 0;
}
