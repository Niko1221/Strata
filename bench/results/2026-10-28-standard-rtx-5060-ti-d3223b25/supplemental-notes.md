# Supplemental notes

These notes were added by hand after `strata bench run` wrote the report. They do not change `README.md`,
`results.json` or `outputs.jsonl`.

## How the run was made

- The Terminal was updated to 0.2.7 (`install.sh` from the 0.2.7 release, checksum and signature checked) and the engine
  to 0.1.52 `linux-x86_64-cuda-cuda13` (`strata update`, signed feed sequence 115) on the same day, just before the run.
- The server was started with `strata launch --preset "IQ3_S 256K tuned" --no-tui`. That is the setup-generated
  IQ3_S 256K settings plus the flags in the track notes. Those flags came from `strata engine calibrate` and a hand
  A/B search on this machine, measured on engine 0.1.48. They were not measured again on 0.1.52.
- The benchmark ran right after the start, against that server, with
  `strata bench run --profile launch-iq3_s-256k-tuned --no-start`. The only request before the run was the launcher's
  one-token readiness check (53 prompt tokens, 1 generated); the report counts it ("answered 1 request before this run").
- The server listens on the local network (`--access lan`) with an API key. The benchmark used loopback.

## Engine log

`--no-start` does not copy the engine log, so `engine.log` was added by hand: the log of this engine session, from its
start line to the end of the run. Home folders and the local network address are replaced with `<HOME>` and `<LAN-IP>`.

## Observations from the log

- Only 388 MiB of VRAM was free after load, and the engine printed its low-VRAM warning (#2065). No paging slowdown was
  seen in this run (decode was 59.9-86.6 tok/s on every measured request; 55.9 tok/s on the excluded warm-up and 29.6 tok/s on the one-token readiness check), but this configuration has little headroom on a
  16 GB card.
- The PLE table `mlock` failed because `ulimit -l` is 8 MB on this machine. The table stays in RAM but is not locked.
- The CPU has no AVX-512; the expert kernels run on AVX2. The i7-14700 is a hybrid CPU (8 P-cores, 12 E-cores); the
  13 pool workers sit on the P-cores (`--pool-affinity auto`).
- The run config points `--expert-profile` at the copy in the Terminal 0.1.49.1 folder (the preset was saved before the
  update). That file has the same SHA-256 (`8f59b4aa8873209d...`) as the copy shipped in Terminal 0.1.52.

## Not measured

- The setup defaults without the tuning flags (no same-day baseline on this engine).
- The optional concurrency stage, long-context (32K and above) prompts, and stability over a long session.
- GPU clocks, temperatures and power during the run. The card's power limit is 180 W (default).
- PCIe link: the card reported Gen 4 x8 at idle.
