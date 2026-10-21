# ram8

- started: 2026-10-21T12:41:47.972976+00:00
- ready in: 52.0 s
- page cache at the start: cold (3.9 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 13939, "disk": ["/dev/sda2", "475G", "433G", "41G"]}
- git: 9ae93f7b
- engine: 0.1.46.1
- config: `config.json` in this folder

## what the engine said when it started

```
{
  "expert_cache_slots": 6160,
  "expert_cache_gib": 6.02,
  "resident_ram_gib": 8.0,
  "pcie_gbs": 12.8,
  "file_cache_hold_pct": 21,
  "direct_read_gbs": 0.56,
  "cached_read_gbs": 0.14,
  "page_cache_gib": 3.2,
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.6,
  "prefill_chunk": 8192,
  "prefill_file_set_gib": 13.7,
  "prefill_page_cache_gib": 3.2,
  "prefill_drive_gb": 14.7
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 3 | 52.0 | 6.1 | 31.66 | 156 | 91.2 | 20397.3 | 14995.6 | 20072.1 | 3251 | 77.4 | 444.7 | 0.02 | 83.44 | 33.11 | 0.05 | 8.02 |
| prompt | 3 | 90.1 | 9.4 | 46.12 | 16 | 93.6 | 14465.0 | 15019.8 | 9420.4 | 2156 | 66.7 | 244.41 | 0.01 | 96.39 | 17.59 | 0.03 | 8.09 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 1 | 90.1 | 9.4 | 46.116 | 15046.2 | 9420.4 | 80.1 | 0.01 | 37.78 | 8.09 |
| decode | 1 | 52.0 | 4.8 | 37.078 | 24217.1 | 26012.8 | 85.0 | 0.45 | 51.34 | 8.02 |
| prompt | 2 | 91.0 | 11.2 | 45.734 | 13133.2 | 8414.3 | 94.0 | 0.02 | 15.54 | 8.2 |
| decode | 2 | 57.1 | 6.1 | 31.658 | 20397.3 | 20072.1 | 91.2 | 0.02 | 33.11 | 9.33 |
| prompt | 3 | 88.6 | 8.8 | 46.931 | 14465.0 | 9745.2 | 93.6 | 0.01 | 17.59 | 7.63 |
| decode | 3 | 51.3 | 9.0 | 29.243 | 15567.1 | 10018.0 | 92.9 | 0.02 | 23.12 | 7.93 |
