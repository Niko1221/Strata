# Expert-profile validation and rollback

`tools/validate_expert_profile.py` checks a candidate expert profile against independent prompts before keeping it. It complements static routing-coverage evaluation: static coverage describes which experts a profile selects, while this tool measures actual decode throughput for code and non-code workloads.

The current profile is backed up first. The candidate is installed and evaluated. If any task group loses more than the allowed median decode-throughput regression, the original profile is restored. The backup is retained for inspection in either outcome.

## Input

Use one JSON object per line in the holdout file:

```json
{"prompt":"Implement a segment tree with range sums.","group":"code","max_tokens":256}
{"prompt":"Write a short paragraph about spring.","group":"general","max_tokens":100}
```

Use separate code and general holdouts. Do not include the prompts used to train the candidate profile.

## Run

```sh
python tools/validate_expert_profile.py \
  --profile data/expert-profile.bin \
  --candidate data/expert-profile-candidate.bin \
  --tasks holdout.jsonl \
  --server http://127.0.0.1:8080/v1 \
  --model MODEL_ID \
  --restart-command "systemctl --user restart my-strata-service" \
  --tolerance 0.10 \
  --report validation-report.json
```

`--restart-command` is optional. Supply it when the engine reads the profile only at startup. If the engine supports live profile reloads, it is not needed.

The JSON report contains `accepted` or `rolled_back`, baseline and candidate median speeds for every group, and the groups that exceeded the regression threshold. The default tolerance is 10%. No engine or model defaults are changed by this tool.

A single fast run is not evidence of a stable speedup. PCIe conditions, cache warm-up, prompt length and other workloads can change results. Compare repeated runs and preserve failures and variability when reporting results.

## Tests

```sh
python -m unittest tools.test_validate_expert_profile
```

The tests cover acceptance, rollback, restoration after evaluation failure, backup protection, task validation and median regression calculations. They do not require a GPU or a model download.
