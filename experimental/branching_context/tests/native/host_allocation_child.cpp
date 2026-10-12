// Real host allocation under a parent-controlled RLIMIT_AS; no model or GPU.
#include <iostream>
#include <new>
#include <string>
#include <unistd.h>
#include <vector>

int main() {
    std::cout << "{\"status\":\"ready\",\"pid\":" << getpid() << "}" << std::endl;
    std::string command;
    while (std::getline(std::cin, command)) {
        if (command == "quit") return 0;
        if (command != "allocate") return 2;
        try {
            std::vector<unsigned char> bytes(64 * 1024 * 1024, 71);
            volatile unsigned char* memory = bytes.data();
            memory[bytes.size() - 1] = 81;
            std::cout << "{\"status\":\"allocated\",\"bytes\":" << bytes.size()
                      << ",\"last\":" << int(memory[bytes.size() - 1]) << "}" << std::endl;
        } catch (const std::bad_alloc&) {
            std::cout << "{\"status\":\"bad_alloc\"}" << std::endl;
        }
    }
    return 0;
}
