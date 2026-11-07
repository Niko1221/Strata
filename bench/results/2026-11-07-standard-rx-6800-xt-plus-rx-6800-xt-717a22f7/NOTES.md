# Author notes — 2x RX 6800 XT (x8 + x8), UD-Q4_K_XL (2026-11-07)

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261107T013408Z-717a22f7`.
The numbers in [README.md](README.md) and [results.json](results.json) are the runner's own and are not edited.

This is a follow-up to [#4868](https://github.com/Niko1221/Strata/pull/4868), on the same machine.
Two things changed together since that report: Strata 0.2.10/engine 0.1.56 became 0.2.11/engine 0.1.57,
and the second card moved from the chipset x4 slot to the CPU x8 slot (x8 + x8). It is therefore a
combined engine + topology result, not an isolated engine comparison.

---

## English

### What changed since #4868

| Item | #4868 (2026-11-04) | **This run (2026-11-07)** |
|---|---|---|
| Strata / engine | 0.2.10 / 0.1.56 | **0.2.11 / 0.1.57** |
| Model | UD-Q4_K_XL (103.7 GB) | same |
| RAM | 128 GB (4 x 32), DDR4-3200 | same |
| PCIe | x16 + x4 (27.9 / 6.3 GB/s measured) | **x8 + x8 (14.1 / 14.0 GB/s measured)** |
| Layer split | K=42 (42 / 6 layers) | **K=25 (25 / 23 layers, auto)** |
| Expert cache | 6,220 pairs (~94.6% routed mass) | **6,216 pairs (~95.7% routed mass)** |
| Drafter | DFlash2 | **DFlash2** |
| Context / KV | 32K / int8 | same |
| Sanity | 12/12 | **12/12** |

### Results side by side

| Case | #4868: 0.2.10, x16+x4, K=42 | **This run: 0.2.11, x8+x8, K=25** | Change |
|---|---:|---:|---:|
| short decode | 29.6 tok/s (28.8–32.8) | **34.1 tok/s (30.6–37.4)** | +15% |
| short TTFT | 3.33 s | **4.25 s** | slower |
| 4K prompt | 331.3 tok/s | **226.1 tok/s** | **−32%** |
| 4K decode | 26.4 tok/s | **35.8 tok/s** | **+36%** |
| 4K TTFT | 12.42 s | **18.16 s** | slower |
| 16K prompt | 460.6 tok/s | **354.9 tok/s** | **−23%** |
| 16K decode | 28.5 tok/s | **31.8 tok/s** | **+12%** |
| 16K TTFT | 35.64 s | **46.23 s** | slower |
| 4K total | 31.8 s | **32.4 s** | ~same |
| 16K total | 53.6 s | **62.3 s** | slower |
| Sanity | 12/12 | **12/12** | same |

Cells are medians from the runner; the full min–max and every request are in `README.md` and `results.json`.
This run measured for 8 min 59 s after the model loaded.

### Why decode improved and prompt/TTFT regressed

- **Decode:** x8/x8 doubles the slower card's host link (6.3 → 14.0 GB/s) and lets the split search choose a
  balanced K=25 instead of K=42. Both cards now hold a similar layer range and expert cache (2,174 slots /
  6.35 GiB on CUDA1, 2,018 MiB free after load). Routed-mass coverage rose to 95.7%, and this run has no
  "0 MiB free" / paging warning.
- **Prompt:** the old x16 card (27.9 GB/s) carried most of the prefill. In x8/x8 both cards run at 14.0 GB/s
  and each prompt chunk crosses the split, so 4K prompt falls 331 → 226 tok/s and 16K 461 → 355 tok/s.
  TTFT follows: 12.42 → 18.16 s at 4K, 35.64 → 46.23 s at 16K.
- **Net:** 4K total is roughly unchanged because the faster decode offsets the slower prompt; 16K total is
  slower because the prompt dominates. For coding/agent work, TTFT matters; conversation-cache reuse can hide
  some of it in real sessions, but the standard suite uses fresh prompts.

### Hardware

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 3950X (16C/32T) |
| Motherboard | ASRock X570 Taichi, BIOS P5.80 |
| RAM | 128 GB (4 x 32 GB), DDR4-3200 |
| GPU | 2 x Radeon RX 6800 XT 16 GB (gfx1030) |
| PCIe | **both cards x8 (14.1 / 14.0 GB/s measured host→device); the second card is no longer in the chipset x4 slot** |
| Storage | NVMe (model files on a WDC WDS500G2B0C) |
| OS | Windows 11 Home 10.0.26300 |
| Driver | 32.0.21045.16010 (the runner's hardware block prints 32.0.21045.11001 for the same cards) |

### Configuration

- `ud-q4_k_xl-32k` profile: 32,768-token context, `--kv int8`, `--expert-cache auto`, `--prefill auto`,
  `--vision` (CPU encoder), `--layer-split auto` (K=25), one server slot.
- Drafter: **DFlash2** (`--draft dflash2 --spec 6 --draft-conf 0.35`, `--draft-fallback none`).
- PLE table (26.8 GiB) resident in RAM (`--ple-io ram`); the profile also sets `STRATA_ROUTE_TAIL_SKIP=0`.
- The 0.1.57 engine package used the existing profile's 0.1.56 expert-profile data and vision binary.
  The engine binary and inference path are unmodified; this report adds result files only.

One setup note for other Windows users: 0.2.11's setup still refuses **new** multi-AMD profiles on Windows
("several AMD cards sharing one model (--gpus) is Linux-only for now"). The already-configured profile starts
through `strata bench run --profile` / `strata serve --profile`, which is how this run was made. This is a
setup guard, not an engine limit; no local patch was needed for this run.

### Limitations

- Draft protocol, single pass, no repetitions.
- 6 of the 9 performance requests reached the 512-token cap while still reasoning (thinking, effort xhigh);
  the 16K case has no first-answer time.
- Warm-cache measurement: the expert cache and PLE table were warm; this is not a cold-model run.
- **Engine and PCIe topology changed together.** The decode/prompt differences cannot be attributed to 0.1.57
  alone. The x8/x8 change is the dominant explanation for the balanced K=25 split and the prompt shift.
- Background desktop work was not measured; the quantization revision was not read.
- Run-to-run variance on this machine is visible; treat the medians as one sample, not a precise speedup claim.

---

## 한국어

### 2026-11-04(#4868) 이후 변경

| 항목 | #4868 (2026-11-04) | **이번 실행 (2026-11-07)** |
|---|---|---|
| Strata / 엔진 | 0.2.10 / 0.1.56 | **0.2.11 / 0.1.57** |
| 모델 | UD-Q4_K_XL (103.7 GB) | 동일 |
| RAM | 128 GB (32×4), DDR4-3200 | 동일 |
| PCIe | x16 + x4 (실측 27.9 / 6.3 GB/s) | **x8 + x8 (실측 14.1 / 14.0 GB/s)** |
| 레이어 분할 | K=42 (42 / 6) | **K=25 (25 / 23, auto)** |
| Expert cache | 6,220쌍 (라우팅 질량 94.6%) | **6,216쌍 (95.7%)** |
| 드래프터 | DFlash2 | **DFlash2** |
| 컨텍스트 / KV | 32K / int8 | 동일 |
| 정합성 | 12/12 | **12/12** |

### 결과 비교

| 케이스 | #4868: 0.2.10, x16+x4, K=42 | **이번: 0.2.11, x8+x8, K=25** | 변화 |
|---|---:|---:|---:|
| short 디코드 | 29.6 tok/s | **34.1 tok/s** | +15% |
| short TTFT | 3.33 s | **4.25 s** | 느려짐 |
| 4K 프롬프트 | 331.3 tok/s | **226.1 tok/s** | **−32%** |
| 4K 디코드 | 26.4 tok/s | **35.8 tok/s** | **+36%** |
| 4K TTFT | 12.42 s | **18.16 s** | 느려짐 |
| 16K 프롬프트 | 460.6 tok/s | **354.9 tok/s** | **−23%** |
| 16K 디코드 | 28.5 tok/s | **31.8 tok/s** | **+12%** |
| 16K TTFT | 35.64 s | **46.23 s** | 느려짐 |
| 4K 총 시간 | 31.8 s | **32.4 s** | 거의 동일 |
| 16K 총 시간 | 53.6 s | **62.3 s** | 느려짐 |
| 정합성 | 12/12 | **12/12** | 동일 |

수치는 러너가 기록한 median이며, 전체 min–max와 개별 요청은 `README.md`와 `results.json`에 있습니다.
모델 로드 후 측정 시간은 8분 59초입니다.

### 왜 디코드는 빨라지고 프롬프트/TTFT는 느려졌나

- **디코드:** x8/x8이 느린 카드의 링크를 6.3 → 14.0 GB/s로 끌어올리고, 분할 탐색이 K=42 대신 균형 잡힌
  K=25를 고릅니다. 양쪽 카드가 비슷한 레이어와 expert cache를 갖고(CUDA1 2,174슬롯 / 6.35 GiB, 로드 후
  2,018 MiB 여유), 라우팅 질량 coverage가 95.7%로 올랐습니다. 이번 실행에는 0 MiB / 페이징 경고가 없습니다.
- **프롬프트:** 이전에는 x16 카드(27.9 GB/s)가 prefill 대부분을 담당했습니다. x8/x8에서는 두 카드가 모두
  14.0 GB/s이고 프롬프트 chunk가 분할 경계를 넘기 때문에 4K 331 → 226 tok/s, 16K 461 → 355 tok/s로
  떨어집니다. TTFT도 4K 12.42 → 18.16 s, 16K 35.64 → 46.23 s로 느려집니다.
- **총평:** 4K는 디코드 상승이 프롬프트 하락을 상쇄해 총 시간이 거의 같고, 16K는 프롬프트 비중이 커서
  느려집니다. 코딩/에이전트 작업은 TTFT가 중요하며, 실제 세션에서는 conversation cache 재사용이 일부
  완화하지만 표준 스위트는 매 요청 fresh prompt입니다.

### 하드웨어

| 항목 | 값 |
|---|---|
| CPU | AMD Ryzen 9 3950X (16C/32T) |
| 메인보드 | ASRock X570 Taichi, BIOS P5.80 |
| RAM | 128 GB (32 GB × 4), DDR4-3200 |
| GPU | Radeon RX 6800 XT 16 GB × 2 (gfx1030) |
| PCIe | **두 카드 모두 x8 (실측 14.1 / 14.0 GB/s); 두 번째 카드를 칩셋 x4 슬롯에서 CPU x8 슬롯으로 이동** |
| 저장장치 | NVMe (모델 파일: WDC WDS500G2B0C) |
| OS | Windows 11 Home 10.0.26300 |
| 드라이버 | 32.0.21045.16010 (러너 하드웨어 블록은 같은 카드에 32.0.21045.11001로 표기) |

### 설정

- `ud-q4_k_xl-32k` 프로파일: 32,768 컨텍스트, `--kv int8`, `--expert-cache auto`, `--prefill auto`,
  `--vision`(CPU 인코더), `--layer-split auto`(K=25), 서버 슬롯 1개.
- 드래프터: **DFlash2** (`--draft dflash2 --spec 6 --draft-conf 0.35`, `--draft-fallback none`).
- PLE 테이블(26.8 GiB) RAM 상주(`--ple-io ram`), 프로파일 env에 `STRATA_ROUTE_TAIL_SKIP=0`.
- 0.1.57 엔진 패키지가 기존 프로파일의 0.1.56 expert-profile 데이터와 vision 바이너리를 사용했습니다.
  엔진 바이너리와 추론 경로는 수정하지 않았고, 이 리포트는 결과 파일만 추가합니다.

Windows 사용자 참고: 0.2.11 setup은 **새** 다중 AMD 프로파일을 Windows에서 여전히 거부합니다
("several AMD cards sharing one model (--gpus) is Linux-only for now"). 이미 구성된 프로파일은
`strata bench run --profile` / `strata serve --profile`로 시작되며, 이번 실행도 그 방식입니다.
이것은 setup 가드이며 엔진 한계가 아닙니다. 이번 실행에는 로컬 패치가 필요 없었습니다.

### 한계

- 드래프트 프로토콜, 단일 실행, 반복 없음.
- 성능 요청 9개 중 6개가 512토큰 상한에서 추론 중 잘렸고, 16K 케이스는 first-answer 시간이 없습니다.
- expert cache와 PLE 테이블이 warm인 상태이며, 콜드 모델 측정이 아닙니다.
- **엔진과 PCIe 토폴로지가 함께 바뀌었습니다.** 디코드/프롬프트 변화를 0.1.57 하나의 효과로 볼 수 없습니다.
  x8/x8 변경이 K=25 균형 분할과 프롬프트 변화의 주된 원인입니다.
- 백그라운드 데스크톱 작업은 측정하지 않았고, quantization revision은 읽지 않았습니다.
- 이 머신은 실행 간 편차가 보이므로, median을 하나의 샘플로 보고 정밀한 배수 주장으로 쓰지 않습니다.
