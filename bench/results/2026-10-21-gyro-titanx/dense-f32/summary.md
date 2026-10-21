# dense-f32

- started: 2026-10-21T13:24:39.065650+00:00
- ready in: 44.0 s
- page cache at the start: cold (3.9 s to evict)
- phase timers: on
- machine: {"cpu": "x86_64", "logical_cpus": 12, "platform": "Linux-7.2.2-1-cachyos-x86_64-with-glibc2.44", "gpu": "NVIDIA GeForce GTX TITAN X", "driver": "580.178.04", "gpu_memory_total_mib": 12288, "gpu_memory_used_mib": 14, "ram_total_mib": 15722, "ram_available_mib": 14066, "disk": ["/dev/sda2", "475G", "433G", "40G"]}
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
  "file_cache_hold_pct": 30,
  "direct_read_gbs": 0.55,
  "cached_read_gbs": 0.14,
  "page_cache_gib": 5.0,
  "draft_vram_mib": 802,
  "engine_version": "0.1.46.1",
  "weights_mib": 1466,
  "weights_load_s": 2.6,
  "prefill_chunk": 8192,
  "prefill_file_set_gib": 15.7,
  "prefill_page_cache_gib": 5.0,
  "prefill_drive_gb": 16.8
}
```

| workload | n | prompt_tok_s | decode_tok_s | ttft_s | generated | expert_hit_pct | os_read_mb | expert_read_mb | decode_wait_ms | decode_fetches | draft_accept_pct | ms_per_window | stage_ms | gpu_reach_wait_ms | cpu_pool_ms | commit_emit_ms | draft_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode | 3 | 50.3 | 5.3 | 34.87 | 141 | 90.4 | 22892.9 | 18417.3 | 23976.1 | 3893 | 75.4 | 496.72 | 0.02 | 76.14 | 35.22 | 0.05 | 7.99 |
| prompt | 3 | 96.0 | 8.9 | 43.34 | 16 | 93.3 | 15771.2 | 17273.3 | 8722.3 | 2246 | 66.7 | 255.84 | 0.01 | 91.63 | 17.83 | 0.03 | 8.44 |

## every run

| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | stage ms/window | cpu ms/window | draft ms/window |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt | 1 | 96.0 | 11.4 | 43.338 | 16536.7 | 8420.0 | 81.4 | 0.01 | 45.76 | 8.46 |
| decode | 1 | 46.8 | 4.4 | 43.221 | 25743.0 | 26285.1 | 81.7 | 0.02 | 62.14 | 7.85 |
| prompt | 2 | 95.2 | 8.7 | 43.718 | 15771.2 | 8912.7 | 93.3 | 0.01 | 17.83 | 8.04 |
| decode | 2 | 50.3 | 5.3 | 34.871 | 22892.9 | 23976.1 | 90.4 | 0.03 | 35.22 | 7.99 |
| prompt | 3 | 96.2 | 8.9 | 43.262 | 14893.5 | 8722.3 | 94.0 | 0.01 | 17.02 | 8.44 |
| decode | 3 | 53.0 | 6.0 | 30.292 | 20095.7 | 17884.4 | 91.1 | 0.02 | 34.09 | 8.41 |
