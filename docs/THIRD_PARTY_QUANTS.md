# Third-party dynamic quants (a n-gram table that is not IQ4_NL)

The engine's PLE reader requires `per_layer_token_embd.weight` in IQ4_NL
(`src/kernels/ngram.cpp`); the 90 B row stride of that encoding is built into
the row-gather path. Publishers quantizing the original checkpoint may ship the
table in another encoding - AtomicChat's AD builds use Q5_1 (the model card
notes 6 vs 8.5 bits were tested with negligible KLD difference). Everything
else in such a build is already accepted by the engine: dynamic i-quant
experts, Q8_0 dense matrices, and Q8_0 small projections via `--compat-bf16`
at pack time. The table is the one tensor that needs handling.

Requantize it once, then pack as usual:

    # keep the original shard under a name shard discovery cannot match
    mv <model>-00002-of-000NN.gguf <model>-00002-of-000NN.gguf.q5_1.bak
    python tools/requant_ple_iq4nl.py \
        --src <model>-00002-of-000NN.gguf.q5_1.bak \
        --out <model>-00002-of-000NN.gguf --workers 16

    python tools/iq_pack.py --gguf <model>-00001-of-000NN.gguf \
        --out <pack-dir> --compat-bf16

- The encoder is a faithful port of ggml's `quantize_row_iq4_nl_impl` (`ntry`
  7, no quant_weights, single-scale branch), including the exact
  `best_index_int8` tie-break. The output decodes identically through a
  canonical IQ4_NL decoder; the run prints RMSE and max abs error against the
  Q5_1 source, and re-verifies head/mid/tail of the written file through the
  canonical decoder.
- The replacement keeps the source shard's header with only the 4-byte tensor
  type field changed, so the table still fills its shard exactly - the engine
  asserts that at load.
- The replacement must keep the original shard's file name: the engine finds
  shards by name.
- `--rows N` limits the run to the first N rows (a short smoke before the full
  table; the full pass on AtomicChat AD-4.27bpw-Q4_K_M-M64 took 79 min for
  320,001,536 rows on 16 workers, RMSE 0.000578).

MTP draft acceptance and prefill/decode on the requantized build behave the
same as on builds whose table is natively IQ4_NL; the acceptance figure moves
with sampling randomness and single requests are not representative (see the
bench notes in the PR that added this tool).
