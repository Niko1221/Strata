<h1 align="center">Strata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · **한국어** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>일반 게이밍 PC에서 1,250억 매개변수 AI 모델 실행</b><br>
NVIDIA 또는 AMD 그래픽 카드(12 GB 이상) · Windows 또는 Linux · 무료 오픈 소스</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Strata 모델이 만든 복셀 탑 정원이 브라우저에서 실행되는 모습"></a><br>
<sub>RTX 5070에서 Strata(IQ3_S, 128K 컨텍스트)에 한 번의 프롬프트로 만든 복셀 탑 정원 ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">전체 영상(49초)</a></sub></p>

Strata는 일반 PC에서 **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**를 실행합니다.
보통 서버가 필요한 크고 성능 좋은 AI 모델입니다. 대화하고, 코드를 작성하고, 이미지를 읽고, 앱 및 코딩 에이전트와
함께 작업할 수 있습니다. 데이터는 PC 밖으로 나가지 않습니다.

## 얼마나 빠른가요?

일반 게이밍 PC 두 대에서 측정했습니다. 영어 기준으로 토큰 하나는 단어 약 ¾개에 해당합니다.

- **답변 생성:** 짧은 대화에서 답변이 나타나는 속도입니다. 초당 60 토큰이면 사람이 읽는 속도보다 빠릅니다.
- **프롬프트 읽기:** 보낸 내용을 읽는 속도입니다(여기서는 문서, 코드 또는 대화 기록 32K 토큰).

<table>
<tr><th>NVIDIA: RTX 5070(12 GB), Ryzen 5 7600, RAM 64 GB</th><th>AMD: RX 9070 XT(16 GB), Ryzen 9 3900X, RAM 47 GB</th></tr>
<tr><td>

| 크기 | 답변 생성 | 프롬프트 읽기 |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| 크기 | 답변 생성 | 프롬프트 읽기 |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA 측정은 Q2_0에 엔진 0.1.36, 나머지에 0.1.26을 사용했습니다(답변 4K, 프롬프트 32K). 전체 표는
[DETAILS.md](docs/DETAILS.md#speed-measured)에 있습니다. VRAM이 큰 카드는 더 빠릅니다. RTX 3090(24 GB)은
초당 약 100~140 토큰을 생성할 것으로 예상됩니다. 긴 대화와 다른 카드의 속도는 [모델별 속도](docs/MODELS.ko.md#크기별-속도)와
[커뮤니티 측정 결과](docs/COMMUNITY_BENCHMARKS.md)를 참고하세요.

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata는 무료입니다. PC에서 잘 실행된다면 커피 한 잔으로 개발을 계속할 수 있게 도와주세요.</sub></p>

## 필요한 사양

| | |
| --- | --- |
| **그래픽 카드** | **NVIDIA** GeForce RTX 20, 30, 40 또는 50 시리즈, 또는 **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 또는 RX 6800 / 6900 시리즈. **VRAM 12 GB 이상**이 필요합니다. |
| **RAM** | 32 GB 이상. RAM 용량에 따라 [실행할 수 있는 모델](#어떤-모델을-선택해야-하나요)이 달라집니다. 64 GB면 모든 크기를 실행할 수 있습니다. |
| **디스크** | 약 80 GB의 여유 공간. 가능하면 SSD를 사용하세요. 처음 시작할 때 훨씬 빠릅니다. |
| **시스템** | Windows 10 / 11 또는 Linux와 NVIDIA 또는 AMD의 최신 그래픽 드라이버. |

나머지는 설치 프로그램이 모두 설정합니다. 그래픽 카드 두세 장에 모델을 나누어 올릴 수도 있습니다([멀티 GPU](docs/MULTI_GPU.md)).

다음 환경은 커뮤니티 구성원이 자신의 컴퓨터에서 작성하고 테스트한 실험적 지원입니다.

- **구형 그래픽 카드**(Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [구형 GPU](docs/OLDER_GPUS.md).
- **Intel Arc**(Linux에서 소스 빌드): [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max(Strix Halo)**(Linux에서 소스 빌드): [Strix Halo](docs/STRIX_HALO.md).
- **AVX2를 지원하지 않는 구형 프로세서:** 작동하지만 느립니다. [구형 CPU](docs/INSTALL.ko.md#구형-cpu실험적).

전체 목록은 [설치 문서](docs/INSTALL.ko.md#필요한-사양)에 있습니다.

## 설치

### AI에게 설정 맡기기

AI 코딩 도우미(Claude Code, Cursor, Codex, GitHub Copilot 등)를 사용하나요? 다음 문장을 붙여 넣으세요.

```text
이 PC에 Strata를 설정해 줘: https://github.com/Niko1221/Strata - 저장소의 docs/AI_SETUP.md를 따라 진행해.
```

그래픽 카드, RAM, 디스크를 확인하고 알맞은 모델을 고릅니다. 그런 다음 설치하고 실행한 뒤 앱을 연결하는 방법을
알려 줍니다. AI 도구는 Strata의 [MCP 서버](docs/MCP_SERVER.md)를 통해 Strata를 설치하고 시작하거나 중지할 수도 있습니다.

### 직접 설치하기

[Strata를 다운로드](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip)하고 압축을 풉니다(또는 `git clone`).
**Windows:** **`START-HERE.bat`**을 두 번 클릭합니다. **Linux:** Strata 폴더에서 **`./setup.sh`**를 실행합니다.

NVIDIA와 AMD 모두 설치 단계는 같습니다. 설치 프로그램이 그래픽 카드를 찾아 알맞은 엔진을 설정합니다. 몇 가지를 묻습니다.

- 사용할 모델과 크기
- 컨텍스트 크기(모델이 기억할 텍스트의 양)
- 이미지 읽기 사용 여부

질문마다 Enter 키를 누르면 권장 답변을 선택합니다. 그러면 모델(약 70 GB)을 다운로드하고 실행합니다. 다운로드가
중단되면 다시 실행하세요. 중단된 곳부터 이어집니다. 브라우저에서 `http://127.0.0.1:8080`의 Strata 앱이 열립니다.

> **모델이 시작되는 동안 PC가 1~3분간 느려지거나 응답하지 않을 수 있습니다**(처음 실행할 때 가장 오래 걸림).
> Strata는 RAM에 35~55 GB를 불러오고 그중 일부를 그래픽 카드용으로 고정합니다. 정상적인 동작이므로 기다리고 창을
> 닫지 마세요. 창에서 Strata가 하는 일을 확인할 수 있습니다.

**다음부터는** `START-HERE.bat`(또는 `./setup.sh`)을 다시 실행하세요. 바로 시작하며 같은 파일을 다시 다운로드하지
않습니다. 창을 닫으면 모델이 중지됩니다. `UPDATE.bat`(`./update.sh`)은 Strata를 시작하지 않고 업데이트합니다. 업데이트,
Docker, 여러 그래픽 카드, 파일 저장 위치와 모든 옵션은 [설치 문서](docs/INSTALL.ko.md)에 있습니다.

**내 PC에서는 얼마나 빠를까요?** `BENCH.bat`(`./bench.sh`)을 실행하세요. 약 5~10분 동안 현재 설정으로 대화,
4K/16K 프롬프트와 에이전트 방식 세션을 측정하고 공유할 수 있는 보고서를 작성합니다([방법](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

## 어떤 모델을 선택해야 하나요?

설치 프로그램이 RAM에 맞는 모델을 추천합니다. 같은 모델도 압축 정도가 다른 여러 크기로 제공됩니다. 작은 크기는
빠르고, 큰 크기는 조금 더 똑똑합니다.

| RAM | 선택 | 이유 |
| --- | --- | --- |
| **32 GB** | **Coder** | 32 GB에 들어가며 코드 작업용입니다(24 GB 그래픽 카드가 있으면 Q2_0과 IQ2_XS도 실행 가능). |
| **48 GB** | **IQ2_XS**(또는 가장 빠른 Q2_0) | 더 큰 크기는 들어가지 않습니다. |
| **64 GB** | **IQ2_XS**(권장), 또는 IQ3_XXS / IQ3_S | 모든 크기를 실행할 수 있습니다. IQ3_S가 가장 뛰어나지만 가장 느립니다. |
| **96 GB 이상** | **IQ3_S**, 또는 Unsloth의 UD-IQ4_XS(약 4비트) | 다른 프로그램을 열어 둔 채로 가장 큰 크기를 실행할 여유가 있습니다. |

- **[Coder](docs/MODELS.ko.md#coder):** 전문가 절반을 제거한 코딩 버전입니다. 제작자가 측정한 전체 모델 SWE-bench Verified
  점수의 91%에 도달하며 RAM 32 GB에 들어갑니다. 중국어와 다른 CJK 텍스트를 포함해 코드 외 영역에서는 성능이
  떨어집니다(#438). 이 용도에는 모든 전문가를 유지하는 Q2_0, IQ2_XS 또는 IQ3_S를 선택하세요.
- **[Swift 1.5](docs/MODELS.ko.md#swift-15):** 답하기 전 생각하는 시간을 크게 줄인 미세 조정 모델입니다. 비슷한 품질의 답변을
  더 빨리 받을 수 있습니다.
- **[Unsloth UD-IQ4_XS](docs/MODELS.ko.md#unsloth-ud-iq4_xs):** Unsloth의 약 4비트 버전으로 품질은 IQ3_S와
  UD-Q4_K_XL 사이입니다. 다운로드 크기는 94 GB입니다. RAM이 약 80 GB보다 적으면 답변하는 동안 SSD에서 일부를 읽으므로
  더 느립니다(NVMe SSD를 사용하면 도움이 됩니다).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.ko.md#unsloth-ud-q4_k_xl-실험적)**(실험적): 전체 모델에 가장 가깝습니다.
  하지만 답변하는 동안 SSD에서 대부분을 읽기 때문에 RAM 64 GB PC에서는 초당 7~8.5 토큰만 생성합니다.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.ko.md#orcarouter-uncensored-iq3_xxs):** 수동으로 설정해야 하며
  설치 프로그램 메뉴에는 없습니다.

크기, 다운로드와 각 사양에 맞는 모델은 [모델 문서](docs/MODELS.ko.md)를 참고하세요. 나중에 다른 모델을 추가하려면
`SETUP.bat`(Linux: `./setup.sh --setup`)을 실행하세요.

## 사용하기

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="코딩 에이전트 옆에 표시된 Strata 앱의 모니터 탭"><br>
<sub>코딩 에이전트가 영상 속 탑 정원을 만드는 동안 Strata 앱의 <b>모니터</b>(왼쪽)를 표시한 모습</sub></p>

- **브라우저에서:** `http://127.0.0.1:8080`을 여세요. **Chat**, 모델과 GPU/CPU/RAM 상태를 실시간으로 보여 주는
  **Monitor**, 설정과 주소를 보여 주는 **About**이 있습니다.
- **앱과 코딩 에이전트에서:** 기본 URL이 **`http://127.0.0.1:8080/v1`**인 "OpenAI 호환" 제공자를 추가하세요.
  API 키와 모델 이름은 아무 값이나 사용할 수 있습니다.
  - Anthropic API를 사용하는 앱: `http://127.0.0.1:8080/v1/messages`(Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI와 OpenAI Responses API를 사용하는 다른 앱: `/v1/responses`
    ([설정](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **사고 수준:** 대화 메뉴나 앱의 "reasoning effort"에서 **off, low, medium, high** 중 하나를 선택하세요. Off가 가장
  빠르고 어려운 질문에는 High가 가장 좋습니다.
- **이미지:** 설정 중 "Images?" 질문에 yes로 답하세요. 그런 다음 대화에서 **Picture**를 클릭하거나 앱에 이미지를
  첨부하세요. AMD 카드는 Linux에서 프로세서를 통해 이미지를 읽을 수 있지만 Windows에서는 아직 읽을 수 없습니다.
- **휴대전화 또는 다른 PC에서:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. 항상 키를 설정하세요.
- **한 번에 요청 하나:** 기본적으로 Strata는 요청 하나에 답하고 나머지는 기다립니다. 여러 요청에 동시에 답하려면
  `"parallel": 2`를 설정하세요([BATCHING.md](docs/BATCHING.md)). 12 GB 카드에서는 각 답변이 더 느려집니다.
- **긴 프롬프트:** Strata는 대화의 첫 메시지를 전부 읽으며, 30,000 토큰당 약 1분이 걸립니다. 후속 메시지는 몇 초 안에
  답변을 시작합니다.

더 알아보기: [대화 저장 위치](docs/INSTALL.ko.md#저장-위치), [API](docs/DETAILS.md#using-it).

## 문제가 생겼나요?

- **Strata를 처음 시작할 때 PC가 멈췄습니다.** 모델을 불러오는 동안에는 정상입니다. 기다리고 창을 닫지 마세요.
  10분이 지나도 멈춰 있다면 PC를 다시 시작하고 다른 프로그램을 닫은 뒤 다시 시도하거나 더 작은 크기를 선택하세요.
- **다운로드 또는 설치 도중 중단됐습니다.** `START-HERE.bat`(또는 `./setup.sh`)을 다시 실행하세요. 중단된 곳부터 이어집니다.
- **매우 느리고 디스크 표시등이 계속 깜박이거나 "the engine stopped unexpectedly"라는 메시지가 나옵니다.** 사용 가능한
  RAM이 부족합니다. 다른 프로그램(브라우저는 RAM을 많이 사용함)을 닫거나 더 작은 크기(Q2_0 또는 IQ2_XS)를 선택하세요.
- **8080 포트가 이미 사용 중이라고 나옵니다.** Strata가 이미 실행 중입니다. 실행 중인 창을 찾으세요.

더 많은 문제와 해결 방법은 [문제 해결 문서](docs/TROUBLESHOOTING.ko.md)에 있습니다. 그래도 해결되지 않으면
[이슈](https://github.com/Niko1221/Strata/issues)를 열고 Strata 폴더의 `strata-<model>.log`를 첨부하세요. 보안 문제를
발견했다면 [SECURITY.md](SECURITY.md)에 따라 비공개로 신고하세요.

## 어떻게 작동하나요?

이와 같은 모델은 보통 수백 GB의 그래픽 메모리가 있는 서버에서 실행됩니다. 일반 그래픽 카드의 메모리는 12~24 GB입니다.
Strata는 **PC 전체에 작업을 나누어** 모델을 실행합니다. 부엌에 비유하면 자주 쓰는 물건은 조리대에 두고 나머지는
식료품 저장실에 두는 것과 같습니다.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="모델의 전문가 24,576개 중 가장 많이 쓰는 전문가는 그래픽 카드에, 전체 전문가는 RAM에, 조회 테이블은 SSD에 저장"></p>

- **모델은 24,576명의 작은 전문가("expert")로 이루어진 팀입니다.** 각 단어에는 그중 10명만 필요합니다.
- **그래픽 카드**에는 가장 자주 쓰는 전문가 수천 명을 둡니다. **RAM**에는 모든 전문가를 저장하고, **프로세서**는
  동시에 나머지를 처리합니다. **SSD**에는 큰 조회 테이블을 저장합니다.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="작은 도우미가 다음 단어를 추측하고 큰 모델이 한 번에 모두 확인해 올바른 단어를 선택"></p>

- **추측하고 확인하기:** 작은 도우미가 다음 몇 단어를 추측합니다. 큰 모델이 한 번에 모두 확인합니다. 같은 답변을
  1.6~1.8배 빨리 받을 수 있습니다.
- **긴 텍스트는 큰 덩어리로 읽습니다**(한 번에 최대 8,192 토큰). 초당 1,000 토큰 이상을 읽습니다.

더 긴 설명은 [작동 원리 문서](docs/HOW_IT_WORKS.ko.md)에 있습니다. 모든 구성 요소와 수치는
[상세 문서](docs/DETAILS.md#how-it-works)와 [논문](docs/paper/Strata-Paper.pdf)을 참고하세요.

## 크레딧과 라이선스

모델은 Qwen 팀의 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)입니다.
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI(Swift 1.5), Unsloth가
모델을 압축했습니다. Strata는 [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)의 일부를 사용합니다. 전체 크레딧은
[작동 원리 문서](docs/HOW_IT_WORKS.ko.md#크레딧)에 있습니다. Strata는 [MIT 라이선스](LICENSE)로 공개된 오픈 소스입니다.
일부 구성 요소와 각 모델에는 별도 라이선스가 적용됩니다([목록](docs/HOW_IT_WORKS.ko.md#라이선스)).

## Strata 후원하기

Strata는 무료 오픈 소스입니다. 유용하게 사용하고 있다면 개발을 후원할 수 있습니다.

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
