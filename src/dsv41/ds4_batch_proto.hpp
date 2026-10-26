// Strata. The engine's side of the server's batch path (docs/DSV41_BATCHING.md, M4).
//
// The server (serve/server.py, generate_batched) drives several requests at once through a batch slot each:
//
//   BGEN  <slot> <max_new> [key=value ...] <id,id,...>        one request admitted into a batch slot
//   BGENI <slot> <max_new> [key=value ...] <pictures> <ids>   the same, with a DSI1 pictures file
//   BSTOP <slot>                                             that slot ends at its next token
//   BYIELD <slot>                                            the control lines are wanted elsewhere: accepted, ignored
//
// An admission is read exactly as `GEN 1` / `GENI 1` would read it (the usual PP lines and the prompt's first token,
// then DONE), and the engine then says `BADM <slot> <1: the slot decodes the rest | 0: the answer is done>`.  When the
// slot goes on its tokens are `BT <slot> <id>`, and the slot ends with `BDONE <slot> <produced> <finish> <ms>` where
// <produced> counts the admission's own token too.
//
// This header is the line bookkeeping with no model in it, so it is checked without a GPU (tests/dsv41/batch_proto_test.cpp).
#pragma once

#include <cstdlib>
#include <string>

namespace strata {
namespace ds4 {

namespace batchproto {

/// A BGEN / BGENI line, split into what the slot keeps and the request the ordinary path reads.
struct Admission {
    int slot = -1;           // the slot the engine was given
    long long max_new = 0;   // what the request asked for, the admission's own token included
    std::string request;     // the same request as a `GEN 1` / `GENI 1` line: the prompt's read and its first token
};

/// BGEN / BGENI -> its slot, its max_new and the GEN 1 / GENI 1 line.  False: not a batch line, or one whose slot and
/// max_new are not both numbers (a malformed line is the server's mistake, and the caller answers ERR).
inline bool split(const std::string& line, Admission& a) {
    const bool img = line.rfind("BGENI ", 0) == 0;
    if (!img && line.rfind("BGEN ", 0) != 0) return false;
    const char* p = line.c_str() + (img ? 6 : 5);
    char* e = nullptr;
    const long b = std::strtol(p, &e, 10);
    if (e == p) return false;
    char* e2 = nullptr;
    const long long mn = std::strtoll(e, &e2, 10);
    if (e2 == e) return false;
    a.slot = (int) b;
    a.max_new = mn;
    a.request = std::string(img ? "GENI 1" : "GEN 1") + e2;
    return true;
}

/// After the admission's DONE: does its slot decode the rest?  Not when the first token ended the answer (EOS), when
/// the request asked for that one token only, or when the read was cancelled: then BADM says 0 and it is over.
inline bool continues(long long produced, long long max_new, const char* finish) {
    return produced == 1 && max_new > 1 && finish != nullptr && std::string(finish) == "length";
}

}  // namespace batchproto
}  // namespace ds4
}  // namespace strata
