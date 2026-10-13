# 어떤 모델을 선택해야 하나요? 크기, 버전과 필요한 사양

[English](MODELS.md) · **한국어**

Strata는 하나의 모델인 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)를 여러 **크기**
(같은 모델을 서로 다른 정도로 압축)와 여러 **버전**(원본, 코딩 버전, 미세 조정 버전)으로 실행합니다.
설치 프로그램이 PC에 맞는 모델을 추천하며, 이 문서는 그 선택을 설명합니다. [README로 돌아가기](../README.ko.md).

> **이 페이지:** [RAM으로 선택](#ram으로-선택) · [속도](#크기별-속도) · [크기](#크기) ·
> [실행 가능한가요?](#실행-가능한가요) · [버전](#버전) · [다른 모델 추가](#모델-추가-또는-전환)

## RAM으로 선택

| RAM | 선택 | 이유 |
| --- | --- | --- |
| **32 GB** | **Coder** | 32 GB에 들어가며 코드에 가장 뛰어나지만 다른 작업에서는 약합니다([이유](#coder)). 24 GB 카드가 있으면 Q2_0과 IQ2_XS도 [저 RAM 모드](#그래픽-카드는-크고-ram은-적은-경우)로 실행되며 일반 용도에는 더 좋은 선택입니다. |
| **48 GB** | **IQ2_XS**(또는 가장 빠른 Q2_0) | 더 큰 크기는 들어가지 않습니다. |
| **64 GB** | **IQ2_XS**(권장), 또는 IQ3_XXS / IQ3_S | 모든 크기를 실행할 수 있습니다(IQ3_S를 사용할 때는 다른 프로그램을 많이 열지 마세요). |
| **96 GB 이상** | **IQ3_S**, 또는 [Unsloth UD-IQ4_XS](#unsloth-ud-iq4_xs)(약 4비트) | 가장 큰 크기를 실행할 여유가 있습니다. |

잘 모르겠다면 **IQ2_XS**를 선택하세요. **Coder**는 RAM 32 GB PC에 들어가는 모델이지만, 코드 데이터로 고른
512개 전문가 중 256개만 유지하므로 코드 외 영역과 영어 이외 언어에서는 성능이 떨어집니다. 일반 용도이거나 RAM이
충분하다면 모든 전문가를 유지하는 모델(Q2_0, IQ2_XS 또는 IQ3_S)을 선택하세요.

## 크기별 속도

RTX 5070(12 GB), Ryzen 5 7600, RAM 64 GB에서 측정했습니다.

| 크기 | 답변 생성(짧은 대화) | 답변 생성(128K 컨텍스트) | 프롬프트 읽기 |
| --- | ---: | ---: | ---: |
| **Q2_0** | 94 tokens/s | 76 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 63 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 49 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 46 tokens/s | 1,620 tokens/s |
| **Coder** (IQ1_M) | 55 tokens/s | 43 tokens/s | 2,180 tokens/s |

AMD RX 9070 XT(16 GB), Ryzen 9 3900X, RAM 47 GB에서 측정했습니다(Linux, 설치 프로그램의 기본 설치.
Q2_0은 설치 프로그램이 계산한 RAM 47 GB 한도를 넘으므로 `--model Q2_0 --yes`로 설치).

| 크기 | 답변 생성(짧은 대화) | 답변 생성(128K 컨텍스트) | 프롬프트 읽기 |
| --- | ---: | ---: | ---: |
| **Q2_0** | 60 tokens/s | 48 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 36 tokens/s | 1,110 tokens/s |
| **Coder** (IQ1_M) | 44 tokens/s | 33 tokens/s | 1,420 tokens/s |

- **답변 생성** = 답변이 나타나는 속도입니다(초당 토큰 수. 토큰 하나는 영어 단어 약 ¾개).
- **프롬프트 읽기** = 보낸 내용(긴 문서, 코드, 대화 기록)을 읽는 속도입니다. 32K 토큰 프롬프트에서 측정했으며,
  4K 프롬프트는 910~1,580 tokens/s로 읽습니다. Q2_0으로 32K 프롬프트를 읽는 데 약 15초가 걸립니다.

VRAM이 큰 카드는 모델을 GPU에 더 많이 올릴 수 있어 더 빠릅니다. RTX 3090(24 GB)은 초당 약 100~140 토큰을
생성할 것으로 예상됩니다. 모든 측정 결과, 긴 컨텍스트 수치와 다른 카드의 예상치는
[상세 문서](DETAILS.md#speed-measured)에 있습니다. AMD 카드별 측정 결과(RX 9070 XT, Radeon AI PRO R9700,
RX 7800 XT, RX 9060 XT, RX 6900 XT)는 [AMD_HIP.md](AMD_HIP.md#rdna4-gfx1201)에 있습니다.

PC마다 결과가 다릅니다. `START-HERE.bat --calibrate`는 현재 PC에서 몇 가지 엔진 설정을 측정해 가장 빠른 설정을
저장합니다(약 15~30분, 느린 카드는 더 오래 걸림. 위 PC에서는 Coder가 7% 빨라짐. 현재는 NVIDIA 카드만 지원).
자신의 PC에서 Strata를 측정했다면 [커뮤니티 벤치마크 결과](COMMUNITY_BENCHMARKS.md)에서 보고서 서식과 pull request로
결과를 공유하는 방법을 확인하세요.

## 크기

| 모델 | RAM+VRAM 요구량 | 속도 | 품질 |
| --- | ---: | --- | --- |
| **Q2_0** | 37.6 GB | 가장 빠름 | 좋음 |
| **IQ2_XS** | 39.2 GB | 빠름 | 더 좋음(**권장**) |
| **IQ3_XXS** | 47.0 GB | 더 느림 | 뛰어남 |
| **IQ3_S** | 54.8 GB | 가장 느림 | 가장 좋음: 공개된 테스트에서 전체 모델과 동일 |

작은 크기 세 개의 다운로드 용량은 66~76 GB입니다([상세 정보](DETAILS.md#which-model)). 처음 시작할 때는
MTP draft layer(약 6 GB, 이미지 사용 시 1 GB 추가)도 다운로드합니다.

## 실행 가능한가요?

Shard 1은 시작할 때 불러오는 모델 부분입니다. 전문가는 **RAM**에, 나머지는 그래픽 카드에 올라갑니다(두 번째 shard인
29 GB 조회 테이블은 SSD에 남음). 따라서 **RAM이 전문가 용량보다 약 10 GB 이상 커야** Windows와 다른 프로그램을
함께 실행할 수 있습니다. 전문가는 shard 1의 대부분을 차지하며 Q2_0은 34 GB, IQ2_XS는 35.5 GB, IQ3_XXS는 43 GB,
IQ3_S는 50 GB, Coder는 23 GB입니다(shard 1의 dense weight는 그래픽 카드에 올라감). RAM 64 GB면 모든 크기를
실행할 수 있고(IQ3_S 사용 시 다른 프로그램을 많이 열지 말 것), 48 GB면 Q2_0과 IQ2_XS를 실행할 수 있습니다.
그래픽 카드가 크면 더 빨라지지만 필요한 RAM이 줄지는 않습니다. 단, 아래 저 RAM 모드는 예외입니다.

### 그래픽 카드는 크고 RAM은 적은 경우

시스템과 함께 전문가를 RAM에 담을 수 없는 PC에서는 설치 프로그램이 모델 파일을 매핑하고 그래픽 카드에 없는
전문가만 RAM에 저장합니다([저 RAM 모드](DETAILS.md#speed-measured), 설치 프로그램이 자동 선택). 예를 들어 RAM 32 GB,
GPU 24 GB PC는 Q2_0, IQ2_XS와 Coder를 이 방식으로 실행하며 RAM 32 GB, GPU 12~16 GB PC는 Coder를 실행합니다.
작은 카드에서는 대부분의 전문가를 SSD에서 읽어 훨씬 느립니다(설치 프로그램이 이를 안내함).
`START-HERE.bat --setup --low-ram on|off`로 선택을 바꿀 수 있습니다.

## 버전

### Qwen3.8-Flash-Next

네 가지 크기로 제공되는 원본 모델입니다. 이미지를 사용할 수 있고,
[실험적 speed projection](DETAILS.md#experimental-speed-projection-experimental-off-by-default)을 선택할 수 있습니다.

### Coder

**[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)**는 ISTA-DASLab의 코딩
버전입니다. 각 layer의 전문가 512개 중 코드(에이전트 및 이미지 포함) 데이터로 고른 256개를 유지하고 나머지를
제거했습니다(제작자 측정 기준 전체 모델 SWE-bench Verified 점수의 91%, LiveCodeBench의 99%). 한 가지 크기만
제공됩니다(IQ1_M, 전문가 저장 방식은 IQ3_S와 같음). Shard 1은 **29.6 GB**이므로 **RAM 32 GB** PC에 들어가고,
64 GB에서는 262K 컨텍스트를 실행하며 모든 모델 중 긴 프롬프트를 가장 빨리 읽습니다. 전문가 절반을 제거했기 때문에
코드 외 영역과 영어 이외 언어에서 약하며 중국어와 다른 CJK 텍스트도 해당합니다(#438: 영어 답변은 정상이었지만
중국어 답변은 잘못되거나 반복됨). 일반 용도이거나 RAM이 충분하면 모든 전문가를 유지하는 Q2_0, IQ2_XS 또는
IQ3_S를 선택하세요. 자세한 내용은 [상세 문서](DETAILS.md#or-the-coder-half-the-experts-for-code)를 참고하세요.

```
START-HERE.bat --setup --family coder
```

### Swift 1.5

**[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**는 UkisAI가 답하기 전
생각하는 시간을 크게 줄이도록 미세 조정한 모델입니다. 거의 같은 품질의 답변을 더 빨리 받을 수 있습니다. 토큰당
속도와 RAM 사용량은 같은 크기의 원본과 거의 같습니다. 자체 라이선스가 적용됩니다(모델 페이지 참고).
자세한 내용은 [상세 문서](DETAILS.md#or-swift-15-a-fine-tune-that-thinks-shorter)를 참고하세요.

```
START-HERE.bat --setup --family swift --model IQ2_XS
```

### Unsloth UD-IQ4_XS

**Unsloth UD-IQ4_XS**(약 4비트)는 설치 프로그램 메뉴의 네 번째 버전입니다(`--family unsloth`의 첫 번째 크기).
0.1.39부터 일반 선택지로 제공됩니다. 다운로드 용량은 94 GB, 전문가 용량은 59.5 GB(IQ3_S 및 IQ4_NL)이며 품질은
IQ3_S와 UD-Q4_K_XL 사이입니다. Strata는 RAM 용량에서 24 GB를 뺀 만큼의 전문가를 RAM에 두고 나머지는 답변하는
동안 SSD에서 읽습니다. RAM 64 GB PC에서는 일부를 SSD에서 읽어 더 느립니다(NVMe SSD가 도움 됨). RAM 약 80 GB부터
모든 전문가가 RAM에 남습니다. RAM 48 GB 이상과 엔진 0.1.38 이상이 필요하며 다른 모델처럼 이미지를 선택할 수 있습니다.
자세한 내용은 [UD-IQ4_XS](UNSLOTH_Q4.md#ud-iq4_xs-setup-from-0139-621)를 참고하세요.

```
START-HERE.bat --setup --family unsloth --model UD-IQ4_XS
```

### Unsloth UD-Q4_K_XL (실험적)

**Unsloth 4비트 UD-Q4_K_XL**(실험적)은 같은 family의 두 번째 크기입니다(`--family unsloth`). 전체 모델에
가장 가깝지만 다운로드 용량이 111 GB이고 전문가 77 GB가 RAM에 모두 들어가지 않습니다. Strata는 RAM 용량에서
24 GB를 뺀 만큼을 RAM에 두고 답변 중 나머지를 SSD에서 읽습니다. RAM 64 GB, GPU 12 GB PC에서 7~8.5 tokens/s로
위 모델보다 몇 배 느리며 긴 프롬프트도 느립니다. RAM 48 GB 이상, NVMe SSD와 NVIDIA GPU 한 개가 필요합니다
(이미지는 아직 지원하지 않음). 자세한 내용과 측정 결과는 [UD-Q4_K_XL](UNSLOTH_Q4.md)을 참고하세요.

```
START-HERE.bat --setup --family unsloth --model UD-Q4_K_XL
```

**Unsloth 6비트 UD-Q6_K_XL**(실험적, 설치 프로그램 메뉴에 없음)은 같은 파일에서 전문가당 바이트 수를 늘려 정밀도
손실을 줄인 모델입니다. 전체 용량은 169.2 GB이고 PLE 테이블은 Q8_0입니다(엔진 0.1.38부터 읽을 수 있음). RAM이
많은 컴퓨터에서는 RAM budget 없이 실행됩니다. 수동 설치 방법은 [UD-Q6_K_XL](UNSLOTH_Q6.md)을 참고하세요.

### OrcaRouter Uncensored IQ3_XXS

**OrcaRouter Flash-Next Uncensored IQ3_XXS**는 [수동 호환 설정](ORCA.md)을 참고하세요. 별도 packing 변환이
필요하며 설치 프로그램 메뉴에는 없습니다.

## 모델 추가 또는 전환

언제든 `SETUP.bat`으로 다른 모델을 추가할 수 있습니다(`START-HERE.bat --setup`과 같음. Linux에서는
`./setup.sh --setup`). 모델끼리 공유하는 파일은 다시 다운로드하지 않습니다(Coder는 원본의 shard 2와 vision
encoder를 공유). 모델을 여러 개 설치하면 `START-HERE.bat`이 시작할 모델을 묻습니다. `run-<model>.bat`
(Linux: `run-<model>.sh`)을 실행하면 해당 모델을 바로 시작합니다.
