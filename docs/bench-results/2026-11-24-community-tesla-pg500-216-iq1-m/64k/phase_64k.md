# 64K prefill phase profile — `STRATA_PREFILL_TIMING=1`, arm A (`STRATA_PF_WMMA=1 STRATA_PF_WMMA2=1`)

Profiled on the same build the same day as the A/B triple (harness as in README §Method; the profiled prompt
was 50 835 tokens vs 50 935 in the triple — same generator family). Clock and power evidence for this box is
in README §Hardware; here the load-level proof is the ratio itself: the GPU timeline consumes 99.5 % of wall,
leaving no room for clock-limiting stalls.

Raw engine line (seed 42 run; second seed agrees within 1 %):

```text
strata prefill timing: 50835 tokens, GPU timeline 26292 ms, wall 26429 ms, host staging 101 ms:
embed+steps 135 (0.5%) hc read 306 (1.2%) gdn 2165 (8.2%) qsa proj 858 (3.3%) qsa indexer 1 (0.0%)
qsa select 1368 (5.2%) qsa attn 3812 (14.5%) router+shared 554 (2.1%) host grouping 161 (0.6%)
gather 254 (1.0%) wait copy 427 (1.6%) dequant 325 (1.2%) gemm gate/up 5381 (20.5%) gemm down 2653 (10.1%)
combine 328 (1.2%) ple 655 (2.5%) gdn conv+gates 167 (0.6%) gdn recurrence 1934 (7.4%) gdn out proj 789 (3.0%)
hc write 943 (3.6%) hc read: down+silu 1078 (4.1%) hc read: up 597 (2.3%) hc read: inject 638 (2.4%)
hc read: mix 608 (2.3%) qsa: indexer gemms 63 (0.2%) qsa: norm/rope/kv append/q prep 94 (0.4%)
```

Grouped (GPU timeline = 99.5 % of wall, so the GPU was the limit and not the host, the link, or the clocks;
`gpu_load_64k.csv` samples this prefill at 2 s: SM clock **1,297–1,380 MHz** — 94–100 % of the 1 380 MHz max,
easing as the HBM heats 48→65 °C — with `power.draw` up to 231 W of the 250 W limit; sustained thermal
throttling pins the clock far below max for the whole run, which is not what these samples show):

| Group | Share | Detail |
| --- | --- | --- |
| Expert GEMM (FP16 TC, `PF_WMMA`) | **30.5 %** | gate/up 20.5 %, down 10.1 % |
| Attention (qsa sparse) | **19.7 %** | attn 14.5 %, select 5.2 %, proj 3.3 % |
| Hyper-connections | **~16 %** | read down+silu 4.1, write 3.6, inject 2.4, mix 2.3, up 2.3, norm 1.2 |
| GDN layers | **~11.6 %** | total 8.2 %, recurrence 7.4, out proj 3.0, conv+gates 0.6 |
| PLE / router+shared / combine / embed | ~6.3 % | 2.5 / 2.1 / 1.2 / 0.5 |
| Card↔host expert streaming | **3.2 %** | wait copy 1.6 %, gather 1.0 %, host grouping 0.6 % |
| Dequant | 1.2 % | |

Reading: prefill is GPU-bound with **no dominant phase** — the grouped gate/up GEMM sits well under the
15.3 TF FP16-TC peak (Volta has no `cp.async`, so the SMs stall inside the kernel), and the **×4 PCIe link
costs ~3 %, not the headline**. Consistent with the arm-A/B result — the gains came from the compute path
(`PF_WMMA`/`PF_WMMA2`); the transfer-side knobs (`STRATA_NV_CHUNK=16`, `STRATA_Q8_INPLACE=1`,
`STRATA_GR_V70=1`) measured as noise (−0.1…+0.1 % over `PF_WMMA` alone, same box, 0.1.67-era build).
