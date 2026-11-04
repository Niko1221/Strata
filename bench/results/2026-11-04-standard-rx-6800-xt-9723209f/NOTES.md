# Author notes — 2x RX 6800 XT, UD-Q4_K_XL, 128 GB (2026-11-04)

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261104T060442Z-9723209f`.
The numbers in [README.md](README.md) and [results.json](results.json) are the runner's own and are not edited.

This is the second run from the same machine as `2026-10-31-standard-rx-6800-xt-d3f713b0` (IQ3_XXS, Strata 0.2.9,
96 GB). It is here to show what 4-bit weights do on the same two cards.

---

## 한국어

### 10월 31일 실행과 달라진 점

| 항목 | 2026-10-31 | **2026-11-04 (이번)** |
|---|---|---|
| Strata / 엔진 | 0.2.9 / 0.1.55 | **0.2.10 / 0.1.56** |
| 모델 | IQ3_XXS (70.6 GB) | **UD-Q4_K_XL (103.7 GB)** |
| RAM | 96 GB (16+16+32+32, 2667) | **128 GB (32×4, 3200)** |
| GPU | RX 6800 XT ×2, x16 + x4 | 동일 |
| 컨텍스트 / KV | 32K / int8 | 동일 |
| 레이어 분할 | K=37 | **K=42** |
| expert arena | 42.9 GiB | **71.7 GiB** |
| expert cache | 12,310쌍 (질량 97.6%) | **6,220쌍 (질량 94.6%)** |

### 결과 비교

| 케이스 | IQ3_XXS (0.2.9) | **UD-Q4_K_XL (0.2.10)** |
|---|---:|---:|
| short 디코드 | 49.0 tok/s | **29.6 tok/s** |
| 4K 프롬프트 | 455.3 tok/s | **331.3 tok/s** |
| 4K 디코드 | 47.4 tok/s | **26.4 tok/s** |
| 4K TTFT | 9.05 s | **12.42 s** |
| 16K 프롬프트 | 670.3 tok/s | **460.6 tok/s** |
| 16K 디코드 | 47.6 tok/s | **28.5 tok/s** |
| 16K TTFT | 24.50 s | **35.64 s** |
| 정합성 | 12/12 | **12/12** |

### 왜 느린가

- **expert 캐시가 절반 이하로 줄었습니다.** 같은 16 GB 카드에 Q4 expert(71.7 GiB)를 넣으니 6,220쌍만
  상주합니다(IQ3는 12,310쌍). 이번 실행의 디코드 적중률은 **34~73%**로, IQ3의 78~92%보다 크게 낮습니다.
- 남은 miss는 RAM에서 PCIe로 올라옵니다. x4 카드의 miss 비용은 4.45배로 계산됐고, 그래서 분할도
  K=42(x16 카드에 42층)로 더 치우쳤습니다.
- 프롬프트도 같은 이유로 느립니다(4K 455→331, 16K 670→461).

### 품질 쪽 이득

`docs/QUANTS.md`의 yardstick 기준으로 top-1은 **84.46% → 93.09%**, KL은 **0.202 → 0.056**입니다.
디코드 40% 손실과 맞바꿀 만한지는 쓰는 사람의 선택입니다. 이 머신에서는:
- 품질 우선 → **UD-Q4_K_XL** (현재 설정)
- 균형 → **UD-IQ4_XS** (expert 59.5 GB, 캐시 적중률이 더 높아 약 40 tok/s 예상, top-1 88.91%)

### 이 PC의 특이점 (변경 없음)

- **PCIe x16 + 칩셋 x4**: 실측 27.9 GB/s vs 6.3 GB/s, x4 카드 miss 비용 4.45배.
- **WDDM 다중 GPU**: expert arena 핀 상한 8 GiB, 대형 페이지 거부로 4 KB 페이지.
- PLE 테이블 26.8 GiB는 RAM 상주(`--ple-io ram`), 로딩 중 38초 소요.
- **로컬 패치 1건**: 0.1.56 `setup.py`가 Windows에서 `--gpu 0,1`을 막아("several AMD cards …
  Linux-only for now") 가드를 안내 문구로 바꿨습니다. **엔진 바이너리와 추론 경로는 손대지 않았습니다.**
  이 패치가 없으면 `serve/server.py`에 `"gpu": "0,1"`을 직접 넣어도 동작합니다(둘 다 같은 엔진 인자).

### 한계

- 드래프트 프로토콜, 단일 실행, 반복 없음
- 실행 전 서버가 요청 1개를 이미 처리(러너 기록) — 완전 콜드 아님
- 성능 요청 9개 중 6개가 512토큰 상한에서 추론 중 잘림
- 백그라운드 작업 미측정, 모델 리비전 미확인
- RAM 증설(96→128 GB)과 모델 교체가 함께 일어나, 두 요인을 분리한 측정은 아닙니다.

---

## English

### What changed since 2026-10-31

| Item | 2026-10-31 | **2026-11-04 (this run)** |
|---|---|---|
| Strata / engine | 0.2.9 / 0.1.55 | **0.2.10 / 0.1.56** |
| Model | IQ3_XXS (70.6 GB) | **UD-Q4_K_XL (103.7 GB)** |
| RAM | 96 GB (16+16+32+32, 2667) | **128 GB (4 x 32, 3200)** |
| GPUs | 2 x RX 6800 XT, x16 + x4 | same |
| Context / KV | 32K / int8 | same |
| Layer split | K=37 | **K=42** |
| Expert arena | 42.9 GiB | **71.7 GiB** |
| Expert cache | 12,310 pairs (97.6% mass) | **6,220 pairs (94.6% mass)** |

### Results side by side

| Case | IQ3_XXS (0.2.9) | **UD-Q4_K_XL (0.2.10)** |
|---|---:|---:|
| short decode | 49.0 tok/s | **29.6 tok/s** |
| 4K prompt | 455.3 tok/s | **331.3 tok/s** |
| 4K decode | 47.4 tok/s | **26.4 tok/s** |
| 4K TTFT | 9.05 s | **12.42 s** |
| 16K prompt | 670.3 tok/s | **460.6 tok/s** |
| 16K decode | 47.6 tok/s | **28.5 tok/s** |
| 16K TTFT | 24.50 s | **35.64 s** |
| Sanity | 12/12 | **12/12** |

### Why it is slower

- **The expert cache is less than half the size.** The Q4 experts (71.7 GiB) leave room for 6,220 pairs on the same
  16 GB cards, against 12,310 for IQ3_XXS. Decode hit rate in this run was **34–73%**, well under IQ3's 78–92%.
- The misses come from RAM over PCIe. The x4 card is priced at 4.45x per miss, so the split moved further toward
  the x16 card (K=42).
- Prompt throughput pays the same cost: 455 -> 331 tok/s at 4K, 670 -> 461 at 16K.

### What the quality side gains

On the `docs/QUANTS.md` yardstick, top-1 goes **84.46% -> 93.09%** and KL **0.202 -> 0.056**. On this machine:
- quality first: **UD-Q4_K_XL** (the current setup)
- balance: **UD-IQ4_XS** (59.5 GB of experts, higher cache hit rate, roughly 40 tok/s expected, top-1 88.91%)

### This machine's unusual parts (unchanged)

- **PCIe x16 + chipset x4**: 27.9 GB/s measured against 6.3 GB/s; the x4 card pays 4.45x per expert miss.
- **WDDM with two GPUs**: the expert arena pins at most 8 GiB and large pages are refused, so it runs on 4 KB pages.
- The PLE table (26.8 GiB) is resident in RAM (`--ple-io ram`), locked in 38 s during load.
- **One local patch:** 0.1.56's `setup.py` refuses `--gpu 0,1` on Windows ("several AMD cards ... Linux-only for
  now"); the stop was turned into a note. **The engine binary and the inference path are untouched.** Without the
  patch, `"gpu": "0,1"` in `serve/server.py`'s config produces the same engine arguments.

### Limitations

- Draft protocol, single run, no repetitions.
- The server had answered one request before the run (recorded by the runner), so this is not fully cold.
- 6 of the 9 performance requests reached the 512-token cap while still reasoning.
- Background work not measured; the quantization revision was not read.
- RAM (96 -> 128 GB) and the model changed together, so the two effects are not separated here.
