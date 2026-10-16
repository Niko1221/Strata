// src/nemotron_h/nh_main.cpp - strata-nh: the Nemotron-H engine path on its own (the same entry point `strata`
// switches to for a nemotron_h_moe model).
#include "strata/nemotron_h/engine.hpp"

int main(int argc, char** argv) { return strata::nemotron_h::run(argc, argv); }
