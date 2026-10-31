# Author notes — 2x RX 6800 XT (x16 + x4), IQ3_XXS

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261031T095044Z-d3f713b0`.
Written by the operator of the machine; the numbers above in [README.md](README.md) and
[results.json](results.json) are the runner's own and are not edited.

---

## 한국어

### 이 머신의 특이점

| 항목 | 값 |
|---|---|
| CPU | AMD Ryzen 9 3950X (16C/32T) |
| 메인보드 | ASRock X570 Taichi (BIOS P5.80) |
| RAM | **96 GB = 16+16+32+32**, 혼합 모듈, **DDR4-2667** 고정 |
| GPU | **RX 6800 XT × 2** (16 GB, gfx1030), 드라이버 32.0.21045.16010 |
| **PCIe 구성** | **GPU0 = x16 (실측 28.0 GB/s host→device), GPU1 = x4 (6.3 GB/s)** |
| 저장장치 | NVMe (모델 파일: WDC WDS500G2B0C) |
| OS | Windows 11 Home 10.0.26300 |
| Strata | 0.2.9 / 엔진 0.1.55 hip-rdna2 (release, commit `c62d5e3a`) |

**가장 큰 특징은 두 카드의 링크 폭이 다르다는 점입니다**(x16 + 칩셋 x4). 엔진은 이걸 그대로 반영합니다:
GPU1의 miss 비용을 **4.38배**로 계산하고(`--layer-split auto`), 48개 레이어 중 **37개를 x16 카드에**,
11개+헤드를 x4 카드에 배치합니다(K=37). K=40이 예측 1.3% 차이의 근접 후보였다는 로그도 남습니다.

### 설정
- 모델: Qwen3.8-Flash-Next **IQ3_XXS** (약 70.6 GB), 컨텍스트 **32K**, `--kv int8`
- `--expert-cache auto` → **5334 슬롯 / 8.28 GiB VRAM**, expert arena 33,352 MiB 잠금
- PLE 테이블 26.8 GiB는 RAM에 상주(`--ple-io ram`)
- `--spec 4`, `--prefill auto`, `--vision`(CPU 인코더), `STRATA_RDNA2_SGEMM=1`
- 샘플링은 모델 카드 기본값(temp 1.0, top_p 0.95, top_k 20), 추론 xhigh

### 결과 요약 (요청 21개 + 워밍업 1개, 12/12 정합성 통과)

| 케이스 | 프롬프트 tok/s | 디코드 tok/s | TTFT |
|---|---:|---:|---:|
| short (~80토큰) | – | **49.0** (44.6–54.6) | 1.88 s |
| 4K | **455.3** (453.6–455.9) | 47.4 (45.6–47.5) | 9.05 s |
| 16K | **670.3** (670.3–674.2) | 47.6 (45.1–47.8) | 24.50 s |

- 디코드 중 **expert cache 적중률 78~92%**(중앙값 약 87%), PCIe/타 GPU로 넘어간 비율은 라우팅된 expert의 **1.2~2.9%**
- 4K/16K 요청 9개 중 7개가 512토큰 상한에서 추론 중 잘림(thinking 모델 특성)
- 모델 로드 후 5분 9초 소요

### 사람들이 궁금해할 만한 점

1. **x4 카드가 있어도 쓸 만한가?** — 씁니다. 다만 엔진이 링크 차이를 가격에 반영해 레이어를 비대칭 배치합니다.
   x4 카드는 expert miss 시 4.38배 비싸므로, 캐시 적중률이 곧 성능입니다(여기선 78~92%).
2. **왜 문서의 R9700 수치(63~64 tok/s)보다 낮나?** — RDNA2(gfx1030)는 행렬 코어가 없고, RAM도
   듀얼채널 2667(≈40 GB/s)이라 문서 기준 장비(EPYC 8채널 106 GB/s)보다 CPU 쪽 expert 처리 대역폭이 훨씬 낮습니다.
   실제로 프롬프트 속도가 4K 455 / 16K 670 tok/s로 R9700(1,100~1,320)의 절반 이하입니다.
3. **96 GB로 충분한가?** — IQ3_XXS(arena 42.9 GB)는 충분합니다. 4비트로 올리면
   UD-IQ4_XS는 실사용 가능(점유 ~77 GB), **UD-Q4_K_XL은 ~105 GB가 필요해 128 GB가 있어야** 편합니다.
4. **Windows 특유의 제약** — WDDM 다중 GPU에서는 expert arena 핀 상한이 **8 GiB**이고,
   대형 페이지 할당은 거부(VirtualAlloc 1314)되어 **4 KB 페이지**를 씁니다.
5. **재현성** — `--kv int8`, 32K, `STRATA_RDNA2_SGEMM=1`을 쓴 **custom 트랙**입니다(셋업 기본값 아님).
   벤치마크 직전 서버가 1개 요청을 이미 처리한 상태였고(러너가 기록), 완전 콜드 측정은 아닙니다.

### 한계
- 드래프트 프로토콜, 단일 실행(반복 없음), 백그라운드 작업 미측정(데스크톱 앱 실행 상태)
- `outputs.jsonl`의 답변은 512토큰 상한에서 잘린 것이 대부분입니다(추론 중 종료).
- 이 PC에는 대시보드용 로컬 패치(텔레메트리 표시)가 적용돼 있으나 **추론 경로에는 영향이 없습니다.**

---

## English

### What is unusual about this machine

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 3950X (16C/32T) |
| Motherboard | ASRock X570 Taichi (BIOS P5.80) |
| RAM | **96 GB = 16+16+32+32**, mixed kits, **DDR4-2667** |
| GPU | **2 x RX 6800 XT** (16 GB, gfx1030), driver 32.0.21045.16010 |
| **PCIe** | **GPU0 = x16 (measured 28.0 GB/s host→device), GPU1 = x4 (6.3 GB/s)** |
| Storage | NVMe (model files on a WDC WDS500G2B0C) |
| OS | Windows 11 Home 10.0.26300 |
| Strata | 0.2.9 / engine 0.1.55 hip-rdna2 (release, commit `c62d5e3a`) |

**The interesting part is the asymmetric link:** one card is on the CPU's x16 lanes and the other on a
chipset x4 link. The engine prices this directly — the x4 card's expert-miss cost is **4.38x** higher —
and `--layer-split auto` puts **37 of 48 layers on the x16 card**, with 11 layers plus the head on the x4 card
(K=37; K=40 was a near tie at +1.3% predicted).

### Configuration
- Model: Qwen3.8-Flash-Next **IQ3_XXS** (~70.6 GB), **32K** context, `--kv int8`
- `--expert-cache auto` -> **5334 slots / 8.28 GiB VRAM**; expert arena locked at 33,352 MiB
- PLE table (26.8 GiB) resident in RAM (`--ple-io ram`)
- `--spec 4`, `--prefill auto`, `--vision` (CPU encoder), `STRATA_RDNA2_SGEMM=1`
- Sampling: the model card's defaults (temp 1.0, top_p 0.95, top_k 20), reasoning xhigh

### Results (21 requests + 1 warm-up; 12/12 sanity checks passed)

| Case | Prompt tok/s | Decode tok/s | TTFT |
|---|---:|---:|---:|
| short (~80 tokens) | – | **49.0** (44.6–54.6) | 1.88 s |
| 4K | **455.3** (453.6–455.9) | 47.4 (45.6–47.5) | 9.05 s |
| 16K | **670.3** (670.3–674.2) | 47.6 (45.1–47.8) | 24.50 s |

- Decode expert-cache hit rate **78–92%** (median ~87%); experts served over PCIe or from the other GPU were
  **1.2–2.9%** of routed experts.
- 7 of the 9 4K/16K requests hit the 512-token cap while still reasoning (thinking model).
- 5 min 9 s of measured work after the model loaded.

### Questions people usually ask

1. **Is a second card on x4 worth it?** Yes, but the engine has to price the link: the x4 card pays 4.38x per
   expert miss, so cache hit rate is what keeps decode fast here (78–92% in this run).
2. **Why lower than the R9700 numbers in the docs (63–64 tok/s)?** gfx1030 has no matrix cores, and the RAM is
   dual-channel DDR4-2667 (~40 GB/s) against the 8-channel EPYC machine in the docs (106 GB/s). Prompt throughput
   shows it most: 455 tok/s at 4K and 670 tok/s at 16K here, versus 1,100–1,320 there.
3. **Is 96 GB enough?** For IQ3_XXS (42.9 GB arena), yes. For 4-bit files, UD-IQ4_XS fits (~77 GB footprint)
   but **UD-Q4_K_XL needs ~105 GB and therefore 128 GB** to stay off the SSD.
4. **Windows-specific limits:** under WDDM with two GPUs the expert arena may pin at most **8 GiB**, and large
   pages were refused (VirtualAlloc error 1314), so the arena runs on **4 KB pages**.
5. **Reproducibility:** this is a **custom track** run (`--kv int8`, 32K, `STRATA_RDNA2_SGEMM=1`), not setup
   defaults. The server had already answered 1 request before the run (recorded by the runner), so this is not a
   fully cold-model measurement.

### Limitations
- Draft protocol, a single run (no repetitions), background desktop work not measured.
- Most answers in `outputs.jsonl` are cut at the 512-token cap while reasoning.
- This PC has a local display-only telemetry patch applied; it does not touch the inference path.
