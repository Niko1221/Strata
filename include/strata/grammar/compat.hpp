// include/strata/grammar/compat.hpp - asserts/logging shims for the ported grammar.
#pragma once
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#define GRAMMAR_ASSERT(cond) assert(cond)
#define GRAMMAR_ABORT(msg) throw std::runtime_error(std::string("grammar: ") + (msg))
#define GRAMMAR_LOG_ERROR(...) std::fprintf(stderr, "strata grammar: " __VA_ARGS__)
#define GRAMMAR_DEBUG(...) ((void)0)
