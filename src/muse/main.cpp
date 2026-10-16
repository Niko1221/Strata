#include "strata/muse/model.hpp"
#include <exception>
#include <iostream>
#include <new>
int main(int argc, char** argv) {
    try {
        const auto options = strata::muse::parse_options(argc, argv);
        std::cerr << "Muse Spark experimental bounded-streaming reference; backend=" << options.backend;
        if (options.backend == "cuda") std::cerr << " device=" << options.gpu;
        else std::cerr << " device=host";
        std::cerr << '\n';
        strata::muse::run(options);
        return 0;
    } catch (const std::bad_alloc&) {
        std::cerr << "Muse: out of host memory; reduce context/geometry\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return 1;
}
