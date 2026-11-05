# Reproducing the community speed and recall measurements

Requires Windows, the Strata checkout at the recorded commit and a preloaded model compatible with the operator's configuration. These commands use the **official tools**, with no source patches from this report. Execute from the repository root, using Python 3.12 (`uv run --no-project --python 3.12` on this Windows machine). A new report directory should be used each time; separate runs by engine restart where a cold model comparison matters.

```powershell
# Run while Strata is serving on localhost:8070; do not launch duplicate servers.
uv run --no-project --python 3.12 .\tools\strata_bench.py --url http://127.0.0.1:8070 --runs 3 --long --clients 4 --out D:\Modelli\Benchmark\reasoning-off
uv run --no-project --python 3.12 .\tools\strata_bench.py --url http://127.0.0.1:8070 --runs 3 --long --clients 4 --think --out D:\Modelli\Benchmark\reasoning-on
uv run --no-project --python 3.12 .\tools\needle_bench.py --url http://127.0.0.1:8070 --lengths 32k,64k,128k --depths 10,50,90 --out D:\Modelli\Benchmark\needles.json
```

**Recording conventions:** The original measurement ran in the sequence shown with a single already-running server. The benchmark chooses a calibration-dependent word-list length, yielding recorded (not exactly nominal) prompt tokens; fresh prompt reuse was zero according to its API timing fields. Four clients queue behind 1 configured parallel engine slot. Needle responses are 40-token-cap, greedy, thinking off. Longer runs may consume substantial RAM/VRAM and GPU time.

For a controlled comparison of two model revisions or launch options, run them in separate freshly loaded engine sessions, record all effective settings, temperature, prompt lengths, output caps, model/pack SHA-256 and engine build flags. Don't infer a speed-up from separate unpaired runs.

## Post-run provenance and limits

The model-file checksums included in [`provenance.json`](provenance.json) were collected separately at 2026-11-05 17:23:42 +01:00 **after** this benchmark. To reproduce **exactly**, configure the server to use the listed model assets and capture the effective running process command/config and pack fingerprints, not merely the on-disk folder. The benchmark's original environment summary reported a modified Git tree; the later provenance collection reported a clean tree at the same commit. Neither observation invalidates the other. A new comparison should restart the engine between arms and record the resulting cache state. The reasoning-on arm only allowed thinking; it did not verify emitted reasoning content.
