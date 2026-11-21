#pragma once

#include "collectives.hpp"
#include <memory>

namespace fnx { class Engine; }
namespace fnx::distributed {
class Runtime {
public:
    Runtime(int& argc, char**& argv);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    void initialize();
    void run(Engine& engine);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
