# base-decode-repeat

- started: 2026-10-21T11:59:54.474465+00:00
- ready in: 46.0 s
- page cache at the start: cold (3.7 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 14116, "disk": ["/dev/sda2", "475G", "433G", "41G"]}
- git: 9ae93f7b
- engine: 0.1.46.1
- config: `config.json` in this folder

## what the engine said when it started

```
{
  "expert_cache_slots": 6160,
  "expert_cache_gib": 6.02,
  "resident_ram_gib": 6.0,
  "pcie_gbs": 12.9,
  "file_cache_hold_pct": 29,
  "direct_read_gbs": 0.56,
  "cached_read_gbs": 0.14,
  "page_cache_gib": 4.8,
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.5,
  "prefill_chunk": 8192,
  "prefill_file_set_gib": 15.7,
  "prefill_page_cache_gib": 4.8,
  "prefill_drive_gb": 16.8
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 3 | 43.4 | 9.0 | 37.28 | 153 | 91.5 | 19663.6 | 18012.2 | 12162.7 | 3349 | 69.5 | 295.62 | 0.02 | 82.5 | 27.4 | 0.05 | 7.85 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| decode | 1 | 39.6 | 4.8 | 47.211 | 28065.6 | 26640.0 | 83.7 | 0.04 | 56.61 | 7.85 |
| decode | 2 | 45.8 | 9.0 | 35.258 | 19663.6 | 12162.7 | 91.5 | 0.01 | 27.4 | 7.95 |
| decode | 3 | 43.4 | 9.0 | 37.285 | 18579.7 | 11664.6 | 92.8 | 0.02 | 22.42 | 7.57 |
