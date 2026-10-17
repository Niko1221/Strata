# Reproduction and artifact notes

Use Strata commit `477ac13cbe8e582c20484871ce459d20d82df09d`, a native SYCL Release AOT `bmg-g31` build, the model revision/profile/configuration in README, and the upstream scripts from that commit. API calls below target loopback only. Adjust the port and protected config path to your own install. Do not publish the config or its API key.

With the existing service running and its active config at `active-model.json`, from the Strata checkout and its Python environment:

```sh
python tools/strata_bench.py --config active-model.json --url http://127.0.0.1:8085 --runs 3 --long --out community-output
```

The speed tool reads authentication from the protected config. One arithmetic/tool smoke check and the tool's own warm-ups preceded measurements; model loading was excluded. No service flags were changed for this suite. Native idle unload remained 600 seconds.

The needle tool only accepts its key as an argument, so our wrapper invoked its Python entry point in-process; the real key was not passed in the operating-system process command line. Equivalent invocation:

```python
import json
from pathlib import Path
import sys

sys.path.insert(0, 'tools')
import needle_bench

config = json.loads(Path('active-model.json').read_text())
sys.argv = ['needle_bench.py', '--url', 'http://127.0.0.1:8085',
            '--api-key', config['api_key'], '--lengths', '32k,128k',
            '--depths', '10,50,90', '--out', 'needles.json']
raise SystemExit(needle_bench.main())
```

Files:
- `results.json`: upstream sanitized speed-suite output, including request metrics, counts, cache state and client latency. Original collector fields are retained; RAM and CPU container limits are explained in README.
- `needles.json`: upstream synthetic recall-check output; elapsed seconds are total client time, not decode throughput.
- `engine-log-excerpt.txt`: startup lines from the matching session only; generic path placeholders replace installation/data roots. Not a full production log.
- `community-build-facts.json`: binary/profile/pack hashes and relevant build/service memory settings; no credentials.
- `strata-community-host-facts.json`: allowlisted CPU, memory, kernel/PVE version, container limits and one host-load snapshot; no hostnames, container identifier or network topology.

Local addresses, identities and credentials are excluded. The server bind address/port in the automatic config view have been replaced by placeholders; this is the only manual change to the upstream speed JSON. Model names, revisions and hashes are public reproducibility metadata. No private conversations, real project files, full run configs, models or packs are included.
