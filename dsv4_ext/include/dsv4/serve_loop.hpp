/**
 * @file serve_loop.hpp
 * @brief Resident serve loop: line protocol between Strata's HTTP server and the dsv4 engine.
 *
 * Implements the stdin/stdout line protocol that Strata's `serve/server.py` speaks to drive
 * inference in `dsv4_run --serve`. This is the same protocol used by Strata's native engine,
 * so the HTTP layer and web app sit on top unchanged — only the backend process differs.
 *
 * @par Protocol specification
 *   @code
 *   [Engine → Server]  READY <max_context> stop            (once, after model loads)
 *   [Engine → Server]  INFO key=value ...                  (optional, before READY)
 *   [Server → Engine]  GEN <max_new> [key=value ...] <id,id,...>
 *   [Engine → Server]  PP <read> <total> <ms> <tok_s>      (prompt progress + heartbeat)
 *   [Engine → Server]  T <id>                              (one per generated token)
 *   [Engine → Server]  DONE <generated> <prompt> <ms> <ms> <finish>
 *   [Engine → Server]  ERR <message>                       (on error)
 *   [Server → Engine]  STOP                                (abort current request)
 *   [Server → Engine]  QUIT                                (shutdown engine)
 *   @endcode
 *
 * @par Sampling keys
 *   Strata's spelling: `temperature=`, `top_p=`, `top_k=`, `min_p=`, `penalty_repeat=`,
 *   `penalty_freq=`, `penalty_present=`, `penalty_last_n=`, `seed=`. Absent temperature = greedy,
 *   matching Strata's engine default.
 *
 * @par Integration with serve_dsv4.py
 *   [`tools/serve_dsv4.py`](../../tools/serve_dsv4.py) imports `serve/server.py`, replaces its
 *   `StrataEngine` with a subclass that spawns `dsv4_run --serve`, and calls its `main()`.
 *   The tokenizer is extracted from the GGUF into `serve-work/tokenizer/` using Strata's own
 *   `tools/strata_tokenizer.py`. Nothing under the Strata checkout is edited.
 */
#pragma once

#include "dsv4/model.hpp"

#include <string>
#include <vector>

namespace dsv4 {

/**
 * @brief Run the resident serve loop: read commands on stdin, write responses on stdout.
 *
 * Drives the model's forward pass in response to `GEN` commands from Strata's HTTP server.
 * Handles token-by-token generation with progress reporting (`PP`), per-token output (`T`),
 * and completion reporting (`DONE`). Supports early stopping via `STOP` command and graceful
 * shutdown via `QUIT`.
 *
 * @par Lifecycle
 *   1. Model must already be loaded (called after `Model::load()` succeeds)
 *   2. Prints `READY <max_context> stop` once initialization is complete
 *   3. Enters command loop: reads GEN, processes tokens, prints PP/T/DONE
 *   4. Exits on QUIT or EOF, returning process exit code
 *
 * @par Heartbeat mechanism
 *   The `PP` (prompt progress) message also serves as a heartbeat for the server's silence watchdog.
 *   If no output is received within the expected timeframe, the server considers the connection dead.
 *
 * @param m       Model instance (must be loaded and ready).
 * @param o       Runtime options controlling generation behavior.
 * @param eos_ids  Vector of end-of-sequence token ids; generation ends early on any of these.
 * @return Process exit code (0 = normal shutdown, non-zero on error).
 */
int serve_loop(Model& m, const RunOpts& o, const std::vector<int>& eos_ids);

}  // namespace dsv4
