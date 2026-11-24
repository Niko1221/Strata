#include "sfnx_input.hpp"

#include <sstream>
#include <stdexcept>
#include <thread>

int main() {
    try {
        for (const char* quit : {"QUIT", "QUIT\r", "  QUIT", "\tQUIT", "QUIT ignored"}) {
            sfnx::Input input;
            std::istringstream stream(std::string("GEN 1 2\n") + quit + "\nGEN 1 3\n");
            std::thread reader([&] { input.run(stream); });
            reader.join();
            std::string line;
            long sequence = 0;
            if (!input.next(line, sequence, false) || line != "GEN 1 2" || sequence != 1)
                throw std::runtime_error("QUIT changed an earlier request");
            if (!input.next(line, sequence, false) || sequence != 0 ||
                input.next(line, sequence, false) || !input.closed())
                throw std::runtime_error("QUIT did not close the queued input");
            if (!std::getline(stream, line) || line != "GEN 1 3")
                throw std::runtime_error("Reader read past QUIT instead of stopping without EOF");
        }
        sfnx::Input input;
        std::istringstream stream("QUITX\nQUIT\n");
        input.run(stream);
        std::string line;
        long sequence = 0;
        if (!input.next(line, sequence, false) || line != "QUITX" ||
            !input.next(line, sequence, false) || line != "QUIT" || !input.closed())
            throw std::runtime_error("QUIT must match a complete command token");
        std::cout << "PASS input_quit_join_without_eof cases=6" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << std::endl;
        return 1;
    }
}
