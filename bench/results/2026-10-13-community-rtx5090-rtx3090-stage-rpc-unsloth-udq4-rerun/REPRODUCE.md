# Reproduction

This results-only report does not publish the custom stage-RPC engine. Use an existing compatible deployment,
with matching build identities, verified Unsloth artifacts, the documented 192K/24+24 configuration and no other clients.
Snapshot the consumed tools/docs/source at Strata commit 7695859aa13afb248bdb4d36f9b2e95707db079d;
do not use a modified working tree as the needle corpus. Hashes are in provenance.json.

Run the unchanged speed script **on the coordinator**, so local hardware discovery describes that machine:

```sh
cd /path/to/frozen-upstream
python3 tools/strata_bench.py --config /path/to/engine/strata-rpc.json \
  --url http://127.0.0.1:8090 --runs 3 --long --out /path/to/new-speed-results
```

The worker-side speed collector attaches by SSH and samples both nodes (it never starts/stops inference):

```sh
STRATA_API_URL=http://coordinator:8090 STRATA_REMOTE=user@coordinator \
STRATA_REPO=/path/to/engine STRATA_FIXTURE_SOURCE=/path/to/frozen-upstream \
  python3 measure_rpc.py --out /path/to/new-private-speed-artifacts
```

The quality controller runs **on the worker**, with the engine and frozen workload roots separate;
the frozen workload must also exist at the same path on the coordinator. Use the retained Node version.

```sh
chmod +x quality/codex_launcher.sh
python3 quality/run_quality.py --source /path/to/engine \
  --fixture-source /path/to/frozen-upstream --remote user@coordinator \
  --url http://coordinator:8090 --launcher /path/to/quality/codex_launcher.sh \
  --out /path/to/new-private-quality-artifacts
```

It includes six recall cases and nine coding attempts; do not additionally replay needles into that interval.
For only recall, run tools/needle_bench.py --url http://127.0.0.1:8090 --lengths 32k,128k --depths 10,50,90 --out needles.json.
Offline collector regressions: cd quality && python3 -m unittest test_collector.py.
Wrappers collect private artifacts by default: review/redact before publishing.

Data: requests/tokens are counts; engine durations milliseconds, client durations/TTFT seconds;
rates tokens/s. Device memory uses bytes in node telemetry and MiB for process readings; summary GiB = MiB/1024.
NIC rates MiB/s and byte deltas since first sample, include unrelated interface traffic. Native cumulative
token counters are exported as deltas; expert-cache capacity is instantaneous and operational cache readings
retain native semantics. Telemetry excludes raw request bodies and historical GPU-process command lines.
Raw generated TypeScript bytes/CRLF are preserved and hashes verified. Git whitespace checks use cr-at-eol.
