// Based on Project Maya (https://github.com/mw00/project-maya)
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Licensed under the MIT License.

// src/ik2/ik2_main.cpp - strata-ik2: the Inkling 2 engine path on its own (the same entry point `strata` switches
// to for an inkling2 model).
#include "strata/ik2/engine.hpp"

int main(int argc, char** argv) {
    return strata::ik2::run(argc, argv);
}
