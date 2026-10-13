# Strata는 어떻게 작동하나요?

[English](HOW_IT_WORKS.md) · **한국어**

Qwen3.8-Flash-Next와 같은 모델은 보통 수백 GB의 그래픽 메모리가 있는 서버에서 실행됩니다. 일반 그래픽 카드의
메모리는 12~24 GB입니다. Strata는 **PC 전체에 작업을 나누어** 모델을 실행합니다. 부엌에서 자주 쓰는 물건은
조리대에 두고 나머지는 식료품 저장실에 두는 것과 같은 원리입니다. [README로 돌아가기](../README.ko.md#어떻게-작동하나요).

## 무엇이 어떤 일을 하나요?

<p align="center"><img src="media/how-it-works.svg" width="860" alt="모델의 전문가 24,576개 중 가장 많이 쓰는 전문가는 그래픽 카드에, 전체 전문가는 RAM에, 조회 테이블은 SSD에 저장"></p>

- **모델은 24,576명의 작은 전문가("expert")로 이루어진 팀이며,** 단어 하나를 쓸 때 그중 10명만 필요합니다.
  따라서 모든 전문가를 그래픽 카드에 한꺼번에 올릴 필요가 없습니다.
- **그래픽 카드**는 모든 단어에 필요한 작업을 수행하고, 가장 자주 요청되는 전문가 수천 명을 저장합니다.
  사용하는 동안 어떤 전문가가 자주 쓰이는지 계속 학습합니다.
- **RAM**에는 모든 전문가를 저장합니다. 단어에 그래픽 카드에 없는 전문가가 필요하면 **프로세서**가 그 작업을
  처리합니다. 그래픽 카드와 동시에 일하므로 어느 쪽도 다른 쪽을 기다리지 않습니다.
- **SSD**에는 큰 조회 테이블을 저장합니다. 모델은 단어마다 작은 행 몇 개만 읽습니다.

기술 용어로 설명하면 다음과 같습니다.

- **GPU(VRAM):** attention 및 DeltaNet mixer, gated-residual 가중치, router, shared expert, output head,
  MTP draft layer, KV cache(64K부터는 가장 많이 읽는 부분만 저장하고 나머지는 RAM에서 스트리밍), 그리고 VRAM의
  남은 공간을 가장 많이 쓰는 전문가로 채우는 **expert cache**를 담당합니다(대화하는 동안 대화 내용에 맞게 조정됨).
  GPU 속도보다 VRAM 용량이 더 중요합니다. 1 GB가 늘 때마다 전문가 약 700명을 더 저장하며, GPU에 있는 전문가는
  CPU가 계산하지 않아도 됩니다.
- **RAM:** 전문가 24,576명을 모두 고정합니다. CPU는 GPU에 없는 전문가를 **제자리에서** 계산하며, GPU는 캐시에
  있는 전문가를 동시에 처리합니다(AVX-512 / AVX2 커널, i-quant에는 ggml 커널). RAM이 적고 그래픽 카드가 큰 PC에서는
  [저 RAM 모드](MODELS.ko.md#그래픽-카드는-크고-ram은-적은-경우)가 GPU에 없는 전문가만 RAM에 저장합니다.
- **SSD:** 28.8 GB n-gram 테이블을 저장하며, 토큰마다 OS 캐시를 통해 행 몇 개를 읽습니다.

NVIDIA(CUDA)와 AMD(HIP) 카드 모두 같은 방식으로 작동합니다. 같은 엔진을 각 환경에 맞게 컴파일합니다.

## 추측한 뒤 확인하기

<p align="center"><img src="media/guess-and-check.svg" width="860" alt="작은 도우미가 다음 단어를 추측하고 큰 모델이 한 번에 모두 확인해 올바른 단어를 선택"></p>

- **추측한 뒤 확인합니다.** 모델에 내장된 작고 빠른 도우미가 다음 몇 단어를 추측하면 큰 모델이 모든 추측을 한 번에
  확인합니다. 동의하는 단어는 유지하고 다음 단어는 직접 씁니다. 따라서 한 단계에서 여러 단어가 생성되는 경우가
  많습니다. 도우미는 추측만 하고 모든 단어는 큰 모델이 결정하므로 같은 품질의 답변을 1.6~1.8배 빨리 얻습니다.
- 기술 용어로는 모델 자체 MTP layer가 최대 3개 토큰을 초안으로 만들고, 48개 layer를 한 번 통과하며 이를 확인합니다.
  통과 한 번당 평균 2.4~3.2개 토큰이 생성됩니다. 답변이 컨텍스트를 반복하는 경우(코드 수정, 인용문)에는
  **prompt lookup**이 앞부분에서 최대 5개 토큰을 초안으로 가져옵니다. 측정한 수락률과 비용상 이득이 있을 때만
  작동하므로 코드 수정은 6~11% 빨라지고 다른 텍스트는 달라지지 않습니다. MTP 초안과 같은 방식으로 확인하므로
  출력도 같습니다.

## 긴 텍스트 읽기

**긴 텍스트는 큰 덩어리로 읽습니다**(단어 조각인 토큰을 한 번에 최대 8,192개). 따라서 긴 문서나 코드베이스를
초당 1,000 토큰 이상으로 읽습니다. 현재 layer의 attention이 실행되는 동안 다음 layer의 전문가가 PCIe를 통해 GPU로
스트리밍됩니다. 첫 메시지를 읽은 뒤에는 대화를 유지하고 새로 추가된 내용만 읽으므로 후속 메시지는 몇 초 안에 시작됩니다.

## 더 알아보기

[상세 문서](DETAILS.md#how-it-works)는 각 구성 요소와 수치를 설명하며,
[논문](paper/Strata-Paper.pdf)에는 이를 뒷받침하는 측정 결과와 전체 원리가 담겨 있습니다.

## 크레딧

- 모델: Qwen 팀의 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)의 압축 버전,
  UkisAI의 [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Unsloth의 실험적
  [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)(지원은
  [eddoursul/Strata](https://github.com/eddoursul/Strata)를 따름). 모델 파일에는 각 모델의 라이선스가 적용됩니다.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)의 일부를 사용해 만들었습니다(MIT).
  [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer),
  [HyperQwen](https://github.com/syv-ai/HyperQwen)에서 아이디어를 얻었습니다. 자세한 내용은
  [상세 문서](DETAILS.md#credits-and-licenses)에 있습니다.
- AMD(HIP) 백엔드의 기여자와 검증에 사용한 컴퓨터는 [AMD_HIP.md](AMD_HIP.md)에 정리되어 있습니다.

## 라이선스

Strata는 [MIT 라이선스](../LICENSE)로 공개된 오픈 소스입니다. 일부에는 별도 라이선스가 적용됩니다.
`third_party/ggml`(MIT, llama.cpp / ggml), 웹 앱 글꼴(SIL Open Font License 1.1),
`data/experimental-speed-projection`의 실험적 speed projection 벡터(Qwen Community License 1.0, 모델의
activation에서 생성)가 해당합니다. 모델은 이 저장소에 포함되지 않으며 각 모델 파일에는 자체 라이선스가 적용됩니다.
