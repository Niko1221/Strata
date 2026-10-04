// dsv4/serve_loop.hpp - the resident serve loop: the line protocol Strata's serve/server.py speaks.
//
//   -> READY <max_context> stop            (once, after INFO lines, when the model is loaded)
//   -> INFO key=value ...                  (optional, before READY: what the Monitor shows)
//   <- GEN <max_new> [key=value ...] <id,id,...>
//   -> PP <read> <total> <ms> <tok_s>      (prompt progress; also a heartbeat for the server's silence watchdog)
//   -> T <id>                              (one per generated token)
//   -> DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish>
//   -> ERR <message>
//   <- STOP                                (end the current request early; it still ends with DONE)
//   <- QUIT                                (exit)
//
// Sampling keys use Strata's spelling: temperature= top_p= top_k= min_p= penalty_repeat= penalty_freq=
// penalty_present= penalty_last_n= seed=.  Absent temperature = greedy, as Strata's engine default.
#pragma once

#include "dsv4/model.hpp"

#include <string>
#include <vector>

namespace dsv4 {

/// Reads commands on stdin and answers on stdout until QUIT or EOF. The model must already be loaded.
/// `eos_ids` end a generation early (the GGUF's tokenizer.ggml.eos_token_id, plus --eos-ids).
/// Returns the process exit code.
int serve_loop(Model& m, const RunOpts& o, const std::vector<int>& eos_ids);

}  // namespace dsv4
