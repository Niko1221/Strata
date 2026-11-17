// STRATA shim: the vendored JSON-Schema→GBNF converter includes llama.cpp's common.h, and only uses
// string_split / string_repeat / string_join from it.  This header (and json_compat.cpp) stands in so the
// converter compiles with zero llama.cpp headers.  The three functions are copied verbatim from
// llama.cpp 3cf03257 common/common.cpp.
#pragma once

#include <string>
#include <vector>

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter);
std::string string_join(const std::vector<std::string> & values, const std::string & separator);
std::string string_repeat(const std::string & str, size_t n);
