# Greedy output across speculative windows

`tools/spec_window_parity.py` compares full output token IDs and finish reasons across repeated requests with
different draft windows. Equal answer text or a plausible opening is not enough. A repeat of the serial reference
must also match. The command exits nonzero on divergence, an incomplete run, an engine error, prompt reuse, or
missing required draft/rollback coverage. It retains every output, including failures.

This is an opt-in diagnostic, not a runtime fix. PR #1773 tests TCP versus an in-process split with no MTP or suffix
drafting. That transport control does not establish draft independence. PR #1748 currently refuses remote
`--pipeline-windows`; test TCP with MTP separately, and add remote overlap only when an engine supports it.

## Run

Prepare a JSON manifest with tokenized prompts and explicit engine arguments. Run private `--serve` engine
processes, not the HTTP server. The script does not modify an existing server, its model configuration, or its
pipeline switch file. Workers needed by a TCP arm must already be listening.

```sh
python tools/spec_window_parity.py --manifest matrix.json --output parity-run
python tools/spec_window_parity.py --audit parity-run/result.json
python -m unittest tools.test_spec_window_parity
```

The output directory must be new. `manifest.json` records the inputs and arguments, `result.json` records full
token IDs and coverage, and `engine-N.log` records stderr. Keep paths and credentials private when publishing
these files: arm environment variables are copied verbatim into the manifest.

Minimal manifest (replace paths and IDs with those for your model):

```json
{
  "repeats": 2,
  "max_new": 128,
  "prompts": [{"name": "code", "ids": [1, 2, 3]}],
  "arms": [{
    "name": "local-mtp",
    "exe": "/path/to/strata",
    "args": ["--pack", "/path/to/pack", "--native", "/path/to/model.gguf",
             "--mtp", "/path/to/mtp", "--spec", "4", "--suffix-draft", "0",
             "--layer-split", "24", "--split-device", "1", "--pipeline-windows", "2",
             "--prompt-cache", "0", "--conversation-cache-mib", "0",
             "--pcie-frac", "0", "--adapt-every", "0"],
    "env": {"STRATA_IQ_MT_MIN": "1", "STRATA_DECODE_TIMING": "1"},
    "cases": [
      {"name": "serial", "keys": {"spec_min_p": 0.5}, "switch": {"pw": 0}},
      {"name": "pipeline", "keys": {"spec_min_p": 0.95}, "switch": {"pw": 2},
       "require": {"drafts": 1, "speculative": 1}},
      {"name": "rollback", "keys": {"spec_min_p": 0}, "switch": {"pw": 2, "force_miss": 1},
       "require": {"speculative": 1, "rollbacks": 1}}
    ]
  }]
}
```

The first case of the first arm supplies each prompt's reference. Cases run in reverse order on alternating
repeats. Only `spec_min_p` is accepted as a per-request key; the request stays greedy. A `switch` uses the engine's
existing diagnostic controls, with `STRATA_PIPELINE_DEBUG=1` and a newly created task-owned file. The gate requires
the engine to confirm the requested `pw` value. `theta=1.1` disables the bonus guess; `force_miss=1` corrupts the
first token of every speculative window to exercise undo. These controls require an engine that implements them.
Coverage counts come from DONE and the `STRATA_DECODE_TIMING=1` pipeline summaries, not from flags alone.

## A useful matrix

Use code, prose, counting, repetitive copying and multi-chunk prompts. Compare draft floors 0, 0.5 and 0.95,
serial versus pipeline, forced rollback and no-guess controls, suffix drafting off versus on, then supported
stage counts and pipeline depths. Separate arms allow different binaries, arguments, backends or transport;
cases within an arm share one process and its initial expert placement. Case names must be unique across arms.
Require nonzero drafts, speculative launches and forced rollbacks where those paths are intended. `suffix_windows`
is also available as a coverage requirement, parsed from the engine's suffix summary. Inspect
offered/accepted draft counts to confirm that changing the draft floor actually changed windows. A suffix-enabled
arm does not by itself prove that suffix drafting supplied a window; require nonzero `suffix_windows`.

First control expert arithmetic and placement: `STRATA_IQ_MT_MIN=1`, no PCIe sharing, no adaptive swaps, identical
resident expert identities, not only slot counts. Disable both prompt and conversation caches. Compare dynamic
adaptation separately; its changes can otherwise obscure a window bug. Record model/pack/profile hashes, engine
binary and source revisions, GPU partition and KV format with the results. The tool hashes each engine binary,
but it cannot establish that two model paths contain identical weights or that resident experts match.

A passing finite matrix supports only its measured models, prompts, stage counts and controls. It does not prove
arbitrary pipeline depth, sampled decoding or general answer quality. A failure identifies the first differing
token (zero-based), including output-length differences, while preserving the complete streams for diagnosis.
