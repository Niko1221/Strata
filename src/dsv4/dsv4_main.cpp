// src/dsv4/dsv4_main.cpp - the standalone DeepSeek V4-Flash executable (strata-dsv4): the same run() the `strata`
// family switch calls for a deepseek4 model, on its own command line.
#include "strata/dsv4/engine.hpp"

int main(int argc, char** argv) { return strata::dsv4::run(argc, argv); }
