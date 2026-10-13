# Strata 설치

[English](INSTALL.md) · **한국어**

NVIDIA 또는 AMD 그래픽 카드를 사용하는 Windows와 Linux에서 Strata를 설치, 시작, 업데이트하고 설정하는 방법을
모두 설명합니다. 짧은 설명은 [README](../README.ko.md#설치)에 있습니다. AI 코딩 도우미가
[AI_SETUP.md](AI_SETUP.md)에 따라 모든 작업을 대신할 수도 있습니다.

> **이 페이지:** [필요한 사양](#필요한-사양) · [Windows](#windows) · [Linux](#linux) ·
> [AMD 카드](#amd-카드) · [여러 카드](#카드-두세-장) · [Docker](#docker-linux) ·
> [구형 CPU](#구형-cpu실험적) · [업데이트](#업데이트) · [저장 위치](#저장-위치) ·
> [설치 프로그램 질문](#설치-프로그램의-질문) · [성능 조정](#내-pc에-맞게-성능-조정) ·
> [전체 옵션](#질문-없이-설정하는-옵션)

## 필요한 사양

| | |
| --- | --- |
| **GPU** | **NVIDIA** RTX 20, 30, 40 또는 50 시리즈, **VRAM 12 GB 이상**(8 GB도 느리지만 실행 가능). RTX 5070과 RTX 3090에서 측정했고 RTX 20(Turing, 0.1.27부터)은 기여자가 RTX 2070에서 테스트했습니다. **AMD** Radeon RX 7900 XT / XTX, RX 9070 / 9070 XT, Radeon AI PRO R9700(검증됨), RX 7800 XT / 7700 XT, RX 9060 XT(소유자가 검증), RX 6800 / 6900 시리즈(커뮤니티 보고), VRAM 12 GB 이상. [AMD 카드](#amd-카드)를 참고하세요. |
| **RAM** | 선택한 크기에 충분한 용량([모델 선택](MODELS.ko.md#ram으로-선택)). **64 GB**면 모든 크기를 실행합니다. 큰 GPU는 부족한 RAM을 보완할 수 있습니다([저 RAM 모드](MODELS.ko.md#그래픽-카드는-크고-ram은-적은-경우)). |
| **CPU** | AVX2를 지원하는 x86-64(최근 약 8년 이내의 Intel/AMD desktop CPU). AVX-512(Ryzen 7000/9000)는 조금 더 빠릅니다. AVX2가 없는 구형 CPU는 실험적이며 느립니다([구형 CPU](#구형-cpu실험적)). |
| **디스크** | 모델에 약 70~80 GB, MTP layer에 약 6 GB(이미지 사용 시 1 GB 추가)의 여유 공간. **AVX-512 CPU의 Q2_0**은 빠른 CPU kernel을 위해 전문가의 일회성 복사본 약 40 GB도 생성합니다. Linux에서 AMD 카드를 사용하면 설치 프로그램이 설치하는 ROCm에 약 10 GB가 더 필요합니다. NVMe SSD를 적극 권장합니다. 처음 시작할 때 훨씬 빠릅니다. |
| **OS** | Windows 10/11 또는 Linux(Ubuntu 22.04/24.04는 필요한 모든 것을 자동 설치). |
| **드라이버** | **NVIDIA:** 580 이상 최신 드라이버([nvidia.com/drivers](https://www.nvidia.com/drivers) 또는 NVIDIA App). **AMD:** Linux에서는 kernel의 amdgpu driver(ROCm 별도 설치 불필요), Windows에서는 최신 AMD Software: Adrenalin Edition driver([amd.com/support](https://www.amd.com/en/support)). |

직접 설치해야 하는 것은 드라이버뿐입니다. 그 밖의 Python, 엔진과 모델은 처음 실행할 때 모두 설정됩니다. Python이
없으면 사용자 계정에 Python 3.12를 설치하고(admin 권한 불필요), `.venv/`에 전용 Python 환경과 Strata 엔진,
모델 및 MTP draft layer를 설치합니다. NVIDIA에서는 RTX 20/30/40/50용 ready-made engine과 pip의 NVIDIA CUDA
library(약 0.4 GB)를 사용합니다. PC에 맞는 ready-made engine이 없으면 build tool 설치를 먼저 물어본 뒤(Windows:
Visual Studio Build Tools + CUDA Toolkit, Ubuntu: `build-essential` + CUDA) 해당 GPU용 엔진을 컴파일합니다
(최초 한 번, 20~40분). 자세한 내용은 [상세 문서](DETAILS.md#before-you-start)를 참고하세요.

## Windows

1. [프로젝트를 다운로드](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip)해 압축을 풉니다(또는
   `git clone`).
2. **`START-HERE.bat`**을 두 번 클릭합니다.
3. 몇 가지 질문에 답하거나 매번 Enter 키를 눌러 권장 설정을 선택합니다([질문 목록](#설치-프로그램의-질문)).

필요한 모든 파일을 다운로드하고(모델은 약 70 GB이므로 처음에는 시간이 걸림. 중단해도 이어받기 가능) **모델을
시작합니다**. 브라우저에서 `http://127.0.0.1:8080`의 Strata 앱이 열립니다.

> **모델이 시작되는 동안 PC가 1~3분간 느려지거나 응답하지 않을 수 있습니다**(처음 실행할 때 가장 오래 걸림).
> Strata는 RAM에 35~55 GB를 불러오고 그중 일부를 그래픽 카드용으로 고정합니다. 정상적인 동작이므로 기다리고 창을
> 닫지 마세요. 창에서 진행 중인 작업을 확인할 수 있습니다.

**다음부터는** `START-HERE.bat`을 다시 두 번 클릭하세요. 모델을 바로 시작하며(불러오는 데 30~90초) 같은 파일을
다시 다운로드하지 않습니다. 창을 닫으면 모델이 중지됩니다. `SETUP.bat`(`START-HERE.bat --setup`과 같음)은 다른
모델을 설치하거나 설정을 변경합니다. 로그인할 때 Task Scheduler로 Strata를 시작하려면 두 가지 task 설정을 바꿔야
하며, 그렇지 않으면 시작 속도가 24배 느립니다([시작할 때 실행](DETAILS.md#running-it-at-startup-task-scheduler)).

## Linux

```bash
./setup.sh
```

같은 질문과 자동 설치를 거쳐(필요하면 Python에 `sudo apt` 사용, 컴파일해야 할 때만 build tool에도 사용)
`http://127.0.0.1:8080`에서 시작합니다. 다음부터 `./setup.sh`(또는 `./run-<model>.sh`)을 실행하면 모델을 바로
시작합니다. `./setup.sh --setup`은 다른 모델을 설치하거나 설정을 변경합니다. 다른 배포판, WSL과 컴파일 방법은
[상세 문서](DETAILS.md#linux)를 참고하세요.

## AMD 카드

NVIDIA와 단계가 같습니다. Windows에서는 `START-HERE.bat`, Linux에서는 `./setup.sh`를 실행하세요. 설치 프로그램이
Radeon 카드를 찾아 Strata에서 사용할 수 있는 NVIDIA 카드가 없는 PC에서는 AMD(HIP) 엔진을 자동으로 선택합니다.
두 종류가 모두 있는 PC에서는 `--backend hip`으로 선택합니다. 지원 카드: RX 7900 XT / XTX(gfx1100),
RX 7800 XT / 7700 XT(gfx1101), RX 9060 XT(gfx1200), RX 9070 / 9070 XT와 Radeon AI PRO R9700(gfx1201),
RX 6800 / 6900 시리즈(gfx1030). 통합 Radeon GPU는 지원하지 않는 것으로 표시됩니다.

Linux에서는 system ROCm 7이 있으면 사용하고, 없으면 AMD wheel에서 `.venv`에 ROCm을 설치합니다(약 10 GB,
sudo 불필요). 그런 다음 해당 카드용 엔진을 PC에서 컴파일합니다(최초 한 번, 10~20분. C++ compiler와 git 필요:
`sudo apt install build-essential git`). NVIDIA와 마찬가지로 AMD 카드 여러 장은 `--gpus`로 모델을 공유합니다.

현재 NVIDIA와 다른 점: 이미지는 CPU의 image encoder가 읽습니다(`--vision cpu`, 8 core에서 기본 image token
300개 기준 이미지당 약 3초). `--calibrate`는 NVIDIA만 지원하며 Unsloth 4비트 모델에는 NVIDIA 카드가 필요합니다.
카드별 측정 결과, 수동 build와 tuning table은 [AMD_HIP.md](AMD_HIP.md)에 있습니다.

## 카드 두세 장

**NVIDIA 카드가 두세 장인가요?** `START-HERE.bat`을 실행하면 카드를 나열하고 Strata에서 사용할 수 있는 카드를
알려 준 뒤 모델을 여러 카드에 나눌지 묻습니다(두 장을 사용할 수 있다면 권장). 카드 한 장에서 만든 설치는 다음 시작
시 한 번 묻습니다. 직접 선택할 수도 있습니다. `START-HERE.bat --gpus 0,2`(두 장, 선택 저장), `--gpus all`, 또는
`--gpu 0`(한 장, 이번 시작에만 적용). 각 카드는 자신이 맡은 layer의 전문가를 저장하고 prompt는 pipeline을 통해
카드를 통과합니다. RTX 5080 + RTX 3090에서는 5080 한 장보다 prompt를 18~20% 빠르게 읽었고 decode 속도는
비슷했습니다. 모든 카드는 VRAM 8 GB 이상의 RTX 20 시리즈 이상이어야 합니다. [MULTI_GPU.md](MULTI_GPU.md)를 참고하세요.

**AMD 카드 여러 장:** 설치 프로그램은 별도로 지정하지 않으면 한 장(VRAM이 가장 크거나 `--gpu N`으로 지정한 카드)을
사용합니다. `--gpus 1,0`은 여러 카드에 layer를 나누며, 설치 프로그램이 표시한 번호 기준 첫 번째 카드가 main card입니다.
카드 한 장에 모델 전문가가 모두 들어가지 않을 때만 분할할 이점이 있습니다([AMD_HIP.md](AMD_HIP.md#rdna4-gfx1201)).

## Docker (Linux)

NVIDIA 카드에서 같은 방식을 container로 실행합니다.

1. Host: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)을
   설치한 Docker와 **580 이상** driver(CUDA 13.0).
2. Build(container 안에서 컴파일하지 않도록 엔진을 image에 컴파일):
   `docker build -t strata .`
   `docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .`는 카드 한 종류만 대상으로 하므로 더 빠릅니다.
   기본값은 RTX 30(86), RTX 40(89), RTX 50(120), A-series(80)를 포함합니다. 이 목록에 없는 카드는 해당 arch로
   다시 build해야 합니다. Image encoder를 빼려면 `--build-arg BUILD_VISION=0`을 추가하세요.
3. 실행(처음에는 약 70 GB 모델을 다운로드한 뒤 시작하고, 다음부터는 바로 serving):
   `docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -v strata-data:/data strata`

   설치 설정은 환경 변수로 전달합니다. `-e MODEL=IQ2_XS -e FAMILY=qwen -e CONTEXT=32768 -e VISION=no`
   (또는 `MODEL=Q2_0|IQ3_XXS|IQ3_S`, `FAMILY=swift|coder`. 위 기본값이 권장 설정).
   `-e VISION=cpu`는 image encoder를 CPU에서 실행합니다. `-e KV=int8|q4_0|k8v4`는 KV cache 정밀도를
   선택합니다. `k8v4`는 INT8 K와 4비트 V를 사용하며 64K부터 KV를 VRAM에 보관합니다.
   `strata-data` volume에는 model file, 준비된 pack, MTP layer와 install config만 저장됩니다. 엔진은 image에
   포함됩니다. Volume에 이미 있는 모델끼리 바꿀 때는 setup을 다시 실행하지 않습니다.
   `-e MODEL=Q2_0 -e FAMILY=coder`는 해당 모델 설정을 선택합니다. 이미 설정한 모델의 설정(context, vision, KV,
   host, api_key, LOW_RAM)을 변경할 때만 `-e REINSTALL=1`을 추가하세요. 설정은 config에 기록됩니다.
   `MODEL`은 `FAMILY`와 함께 `/data/config/strata-<model>.json`을 선택합니다(모델 이름은 소문자). 따라서
   `-e MODEL=IQ3_S.1X4-BATCH9`는 `/data/config/strata-iq3_s.1x4-batch9.json`을 시작하며, 이 방식으로 하나의
   volume에 같은 모델의 config를 여러 개 보관할 수 있습니다. `-e CONFIG=/data/config/my.json`은 임의의 config를
   직접 지정합니다. Entrypoint 실행 전에 pod command가 만든 `/data/config/` 내부를 가리키는
   `/opt/strata/strata-<model>.json` link는 유지됩니다. Entrypoint는 서버가 시작하는 file을 `Config:` 한 줄로 표시합니다.
   Strata는 RAM에 32~62 GB를 불러옵니다. 사용 가능한 카드가 두 장인 host에서 `--gpus all`을 사용하면 둘 다 사용합니다.
   설치 프로그램의 권장 layer split이 기본값이며([MULTI_GPU.md](MULTI_GPU.md)), 카드 한 장용으로 설정한 volume을 두 장
   환경에서 처음 시작하면 두 장으로 전환합니다. `-e GPU=0`으로 한 장을 고정하거나 `-e GPUS=0,2`로 카드를 지정하고
   `-e LAYER_SPLIT=18`로 두 번째 카드의 layer 시작 위치를 정할 수 있습니다.
   Memory limit을 사용하려면 `-e LOW_RAM=on`이 필요합니다. Setup.py는 container limit이 아닌 host RAM을 측정하므로
   제한을 볼 수 없습니다. LOW_RAM은 `GPUS`로 여러 장을 지정하지 않으면 카드 한 장에서 실행됩니다(여러 장이면 카드에
   없는 전문가를 OS file cache로 읽으며 긴 prompt에서 RAM을 채울 수 있음).
   서버는 기본적으로 `0.0.0.0:8080`에서 수신합니다. 네트워크에 port를 공개하기 전에 `-e API_KEY=<secret>`을
   설정하세요. Image에는 `/health`를 확인하는 `HEALTHCHECK`가 있어 모델을 불러온 뒤 `docker ps`에 healthy로 표시되며,
   `GET /v1/status`는 실행 중인 모델을 알려 줍니다.
   이미 가진 model file(설치 프로그램의 `--gguf-dir`)은 해당 폴더를 mount하고 지정하세요. 예:
   `-v /models/UD-Q4_K_XL:/gguf -e GGUF_DIR=/gguf -e FAMILY=unsloth -e MODEL=UD-Q4_K_XL`.
   처음 시작할 때는 writable로 mount하세요. 설치 프로그램이 Unsloth file의 SHA-256을 한 번 확인하고 각 shard 옆에
   `<shard>.done`으로 결과를 저장합니다. 다음부터는 setup을 건너뛰므로 `:ro`로 mount해도 됩니다.
   `-e RESIDENT_BUDGET_GIB=N`과 `-e KV_STREAMING=on|off|auto`는 설치 프로그램의 `--resident-budget-gib`와
   `--kv-streaming`으로 전달됩니다(이미 설정된 모델에는 `REINSTALL=1`도 필요).
   중지할 때 엔진은 SIGTERM을 받으면 page-locked RAM 수십 GB를 해제하므로 Docker 기본 10초보다 오래 걸립니다.
   시간을 더 주지 않으면 `docker stop`이 강제 종료합니다(exit 137).
   `docker run --stop-timeout 60 ...`, `docker stop -t 60 <name>`, 또는 Compose의 `stop_grace_period: 60s`를 사용하세요.

## 구형 CPU(실험적)

Strata ready-made engine에는 AVX2(Intel Haswell 2013, AMD Zen 2017 이상)가 필요합니다. 0.1.39부터 AVX만 지원하는
구형 CPU(Sandy Bridge / Ivy Bridge, Xeon E5 v1/v2, AMD Bulldozer) 또는 SSE4.2만 지원하는 CPU(Nehalem / Westmere,
Xeon X5600)는 설치 프로그램이 해당 PC에서 컴파일하는 **실험적** build로 실행됩니다. 평소처럼 설치 프로그램을
실행하세요. AVX2가 없고 실험적이며 느리다는 경고를 표시한 뒤 엔진을 컴파일합니다(최초 한 번, 10~20분. Windows에는
Visual Studio Build Tools와 CUDA Toolkit이 필요하며 설치 프로그램이 설치할지 물음). 설치를 중단하지 않습니다.
Ready-made engine은 이러한 CPU를 계속 거부하며 이유를 표시합니다.

CPU에 따라 선택되는 `STRATA_ISA_FLOOR=avx` 또는 `none` build의 차이:

- ggml-cpu를 AVX2 대신 AVX 또는 SSE4.2용으로 컴파일하고 CPU가 맡은 전문가는 해당 kernel에서 실행합니다. Strata의
  AVX2 / AVX-512 kernel은 프로그램에 남아 있지만 이를 지원하는 CPU에서만 사용합니다.
- i-quant 모델의 pack만 실행됩니다(해당 CPU에서 설치 프로그램이 설치하는 모든 모델). AVX-512 Q2_0 pack은 실행되지 않습니다.
- 속도: GPU 부분은 같고 CPU 부분은 느리므로 VRAM이 큰 GPU(더 많은 전문가를 cache)가 가장 큰 도움이 됩니다.
  Ryzen 5 7600 + RTX 5070에서 expert cache를 1,500 slot으로 고정하고 구형 path를 강제한 결과, greedy decode는
  Coder와 IQ3_XXS에서 AVX build 사용 시 26 tok/s에서 11~17 tok/s, SSE4.2 build 사용 시 3.5~3.8 tok/s로
  감소했습니다(prompt 10개, 1회 실행). DDR3를 사용하는 구형 Xeon은 이보다 느립니다.
- 실제 구형 CPU에서는 아직 직접 테스트하지 않았습니다. Ryzen 5 7600에서 구형 path를 강제해 확인했습니다
  (`STRATA_FORCE_ISA=avx` 또는 `sse`). 답변은 정상적이었고 엔진 검사를 통과했지만 일반 build와 토큰 단위로 같지는
  않았습니다(ggml-cpu가 CPU expert를 다르게 반올림하며 AVX2 kernel과 AVX-512 kernel 사이에서도 같음). 기여자들은
  이전 버전의 같은 방식을 Xeon E5-2680(GTX 1080 Ti 사용 시 2.87 tok/s)과 Xeon E5-2687W / X5690에서 실행했습니다.
- AMD 카드: Linux에서만 지원(설치 프로그램이 AMD 엔진을 항상 해당 환경에서 컴파일). NVIDIA는 Windows와 Linux 지원.
- CPU socket 두 개인 Xeon: 다른 socket의 memory는 느리므로 가능하면 RAM을 한 CPU에 두세요.

수동 build: `cmake ... -DSTRATA_ISA_FLOOR=avx`(또는 `none`). 어떤 PC에서든 설치 프로그램에
`STRATA_ISA_FLOOR=avx`를 설정하면 이 build를 시험할 수 있습니다. 실제 hardware의 문제와 결과를 GitHub issue로
공유해 주세요.

## 업데이트

**`UPDATE.bat`**(Linux: `./update.sh`)은 모델을 시작하지 않고 Strata를 업데이트합니다. GPU가 다른 일을 하고 있거나
새 버전만 미리 준비할 때 사용합니다. `git clone`에서는 `git pull` 뒤 `START-HERE.bat`이 시작 전에 하는 작업을
실행합니다. 새 버전에 필요한 ready-made engine(또는 source가 변경된 경우 Linux에서 compiled engine 재컴파일),
Python package, 설치된 각 모델의 설정과 draft subset을 업데이트합니다. Model file은 건드리지 않고(새 엔진은 다운로드할
수 있음) 질문도 하지 않습니다. 실행 중인 엔진은 교체할 수 없으므로 먼저 모델 창을 닫으세요. 나중에 평소처럼
`START-HERE.bat`으로 시작합니다. Zip으로 다운로드한 복사본은 새 file을 가져올 수 없으므로 아래처럼 새 zip을 받으라고
안내합니다.

직접 하려면 새 버전을 다운로드해 아무 곳에나 압축을 풉니다(또는 `git pull`). 그런 다음 해당 폴더에서
`START-HERE.bat`(Linux: `./setup.sh`)을 실행하세요. Model file은 Strata 폴더 옆의 `Strata-data` 폴더에 보관되므로
새 복사본도 이를 찾아 같은 방식으로 설정하며 큰 file을 다시 다운로드하지 않습니다. Linux에서는 `git pull` 뒤 source가
바뀌면 엔진을 다시 컴파일합니다(변경된 file은 1~2분). 컴파일에 실패하면 이유를 표시하고 기존 엔진을 시작합니다.

## 저장 위치

- **대화: 브라우저에만 저장됩니다.** Chat tab은 대화, 설정과 입력한 API key를 브라우저 local storage의 `strata.*`
  key에 저장합니다. 서버나 Strata 폴더에는 저장하지 않습니다. 이미지는 이름만 저장하고 내용은 보관하지 않습니다.
  다른 브라우저나 private window에서는 빈 상태로 시작하며 사이트 데이터를 지우면 대화도 삭제됩니다.
- **모델 시작 방법:** Strata 폴더의 `strata-<model>.json`(context, GPU, host, API key 등, 설치 프로그램이 작성),
  옆의 `run-<model>.bat` / `.sh`, `strata-<model>.log`, 그리고 "Use for other apps too"를 사용할 때 생성되는
  `strata-<model>.shared-settings.json`. 같은 모델에 설치 프로그램을 다시 실행하면(예: context 변경) 설치 프로그램이
  작성하는 key(`exe`, `args`, `port`, `gpu`, `host`, `api_key`, `vision` 등)는 다시 쓰고 직접 추가한 key(`sampling`,
  `mcp_servers`, `mcp`, `cors_origins` 등)는 유지합니다. 이전 file은 `strata-<model>.json.bak`으로 보관합니다(0.1.39).
  `"args"`에 직접 추가한 engine option은 이어지지 않습니다. 설치 프로그램이 관리하므로 다시 추가해야 합니다.
- **Model file**(`models/`, `packs/`, `mtp/`, 70~120 GB): **Strata 폴더 옆의 `Strata-data`**, 또는 `--data-dir`로
  지정한 위치.
- **Data folder 위치:** Windows `%APPDATA%\Strata\settings.json`, Linux `~/.config/strata/settings.json`
  ([상세 정보](DETAILS.md#before-you-start)).
- **프로그램 자체:** Strata 폴더의 `.venv/`, `engine/`, `third_party/`.
- **github.com에 접근할 수 없는 환경:** Source build는 `setup.py`의 `LLAMA_CPP_COMMIT`에 지정된 commit에서
  llama.cpp source(ggml, gguf-py)를 GitHub로부터 다운로드합니다. Mirror나 다른 PC에서 해당 commit tree를 받아
  `third_party/llama.cpp`에 두면 설치 프로그램이 사용합니다(#585).

## 설치 프로그램의 질문

- **모델과 크기:** 원본, Swift 1.5, Coder 또는 Unsloth 4비트와 Q2_0, IQ2_XS, IQ3_XXS 또는 IQ3_S.
  [모델 선택](MODELS.ko.md)을 참고하세요.
- **컨텍스트 크기:** 모델이 한 번에 기억할 수 있는 텍스트의 양(카드에 맞는 크기를 제안). 384K와 512K(실험적)는
  rope scaling으로 학습한 262K보다 늘립니다. 설치 프로그램이 자동으로 활성화합니다(yarn과 covering factor,
  `--rope-scaling`/`--rope-scale`로 변경 가능). [상세 문서](DETAILS.md)의 "Context extension past 262K"를 참고하세요.
- **이미지:** 이미지도 읽을지 선택합니다. Image encoder(0.9 GB)는 GPU(이미지당 0.1~0.5초, VRAM 약 1.4 GB를
  비워 둠) 또는 CPU(이미지당 약 3~13초, GPU 사용 없음)에서 실행합니다. AMD 카드는 현재 CPU에서 실행합니다.
  [상세 정보](DETAILS.md#images-vision).
- **실험적 speed projection:** yes를 선택하지 않으면 꺼져 있습니다. 먼저
  [기능 설명](DETAILS.md#experimental-speed-projection-experimental-off-by-default)을 읽으세요. 모델의 답변 방식을
  바꾸며 원본 모델에서만 제공됩니다.

`--yes`를 사용하면 모든 질문에 권장 답변을 선택합니다.

## 내 PC에 맞게 성능 조정

PC마다 결과가 다릅니다. `START-HERE.bat --calibrate`(Linux: `./setup.sh --calibrate`)는 현재 PC의 몇 가지 엔진
설정을 측정해 가장 빠른 설정을 저장합니다(약 15~30분, 느린 카드는 더 오래 걸림. RTX 5070 + Ryzen 5 7600에서
Coder가 7% 빨라짐). 3%보다 빠를 때만 설정을 저장하며 PC와 모델별로 기억하므로 업데이트 뒤에도 유지됩니다. CPU
worker 수를 측정하려면 새 엔진이 필요해 모델을 여러 번 불러옵니다. 다시 시작할 때마다 PC가 바쁘고 1~2분간 응답하지
않을 수 있습니다. 완료되면 **모델을 시작**하므로 이미 서버가 실행 중입니다. 두 번 시작하지 마세요. 현재는 NVIDIA
카드만 지원합니다. [측정 항목](DETAILS.md#double-click-start-herebat).

## 질문 없이 설정하는 옵션

```
START-HERE.bat --setup                          다른 모델 설치 또는 context / image 변경
SETUP.bat                                       위와 같음(두 번 클릭)
START-HERE.bat --model IQ2_XS --context 32768 --vision yes --yes     질문 없음
START-HERE.bat --gguf-dir D:\models\IQ2_XS       가지고 있는 GGUF file 사용
START-HERE.bat --inspect D:\models\some.gguf     GGUF의 실제 내용과 Strata 실행 가능 여부 확인(download 없음)
START-HERE.bat --data-dir E:\Strata-data         model file을 다른 위치에 저장
START-HERE.bat --port 8081                      다른 port
START-HERE.bat --gpu 1                          다른 GPU(설치 프로그램은 VRAM이 가장 큰 카드 선택)
START-HERE.bat --gpus 0,2                       여러 GPU가 모델 공유
START-HERE.bat --vram-reserve-mib 2048          다른 프로그램을 위해 VRAM 2 GB 남김(저장됨)
START-HERE.bat --no-browser                     모델이 준비돼도 chat page를 열지 않음(저장됨,
                                                --browser로 취소)
START-HERE.bat --setup --backend hip            NVIDIA 카드도 있는 PC에서 AMD 엔진 사용
START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>     key를 사용해 다른 device에서 접속
START-HERE.bat --calibrate                      PC에 맞게 엔진 조정(약 15~30분, 느린 카드는 더 오래), 이후 시작
START-HERE.bat --check                          PC만 검사
```

**다른 프로그램을 위한 VRAM 남기기(#493):** Strata는 그래픽 카드의 남은 VRAM을 전문가(expert cache)로 채우고
`--vram-reserve-mib` MiB를 비워 둡니다. 기본값은 700입니다. 게임, 3D 프로그램이나 다른 모델을 함께 사용하려면 더
남겨 두세요. `START-HERE.bat --vram-reserve-mib 2048`(Linux: `./setup.sh --vram-reserve-mib 2048`)은 모델의
`strata-<model>.json`에 값을 쓰고 시작합니다. Setup에서 사용하면(`--setup --vram-reserve-mib 2048`) 새 config에
저장됩니다. 직접 설정하려면 config의 `"args"` list에 `"--vram-reserve-mib", "2048"`을 추가하고 다시 시작하세요.
Expert cache가 그만큼 작아져 답변이 조금 느려질 수 있습니다. 엔진은 시작할 때 비어 있는 VRAM으로 cache 크기를
정합니다. 그때 이미 다른 프로그램이 쓰는 VRAM은 건드리지 않으며 reserve는 나중에 사용할 공간입니다.

**Linux desktop도 표시하는 AMD 카드(#560 #516):** 기본 reserve에서는 expert cache가 카드를 채웁니다. Desktop
(compositor, 브라우저, 새 앱)에 VRAM이 더 필요하면 amdgpu driver가 GPU memory를 system RAM으로 옮기는데 모델
전문가가 이미 RAM을 채우므로 OOM killer가 desktop session(KWin, plasmashell)을 종료하거나 compositor가 실패할 수
있습니다("Failed to pin framebuffer"). 두 보고서에서는 `./setup.sh --vram-reserve-mib 3072`로 해결했습니다
(VRAM의 전문가가 약 2.3 GB 줄고 속도는 몇 % 감소). 설치 프로그램과 서버 창은 해당 PC에서 이를 안내합니다.

**VRAM 8 GB 미만 카드(#496):** 기본 reserve로 expert cache 공간이 너무 작으면 엔진이 들어갈 때까지 reserve를
낮춥니다(최소 300 MiB). 카드가 거의 가득 차면 경고합니다. "no VRAM is left for the expert cache"와 함께 시작이
중단되면 log에 부족한 용량이 표시됩니다. Setup에서 8K context와 `--draft-vocab en`을 사용하면 가장 많은 공간을
확보할 수 있습니다.

**직접 또는 mirror에서 다운로드한 model file(#495):** 설치 프로그램 5단계에서 필요한 폴더를 표시합니다
(`Strata-data\models\<SIZE>\`, 예: `Strata-data\models\IQ3_XXS\`). 원래 이름으로 그 위치에 두거나
`--gguf-dir`로 지정하세요. 사용하거나 다운로드하기 전에 file의 실제 내용을 확인하려면 `--inspect`가 header만 읽습니다
(수 MB. File, folder, URL, ModelScope의 `ms:owner/repo`, Hugging Face의 `hf:owner/repo` 모두 지원). 각 weight group의
실제 bits per weight와 storage type, 설치 프로그램이 설치하는 file과 같은지(이름이 달라도), 같은 방식으로 저장됐는지,
Strata가 실행하지 않는 file인지(다른 architecture 또는 Strata kernel이 읽지 못하는 expert format)를 보여 줍니다.
이름만으로는 알 수 없습니다. Unsloth `UD-IQ3_XXS`의 routed expert는 weight당 3.22비트, `UD-Q4_K_XL`은 5.10비트,
원본 `Q2_0`은 2.25비트, `IQ3_S`는 3.33비트로 저장됩니다. 설치 프로그램이 Hugging Face mirror에서 직접 받게 하려면
먼저 `HF_ENDPOINT`를 설정하세요(Windows: `set HF_ENDPOINT=https://hf-mirror.com`, Linux:
`export HF_ENDPOINT=https://hf-mirror.com`). 같은 고정 revision과 검사를 적용하며 MTP draft layer도 해당 위치에서 받습니다.

**ModelScope 사용(중국 본토):** 설치 프로그램이 다운로드하는 모든 저장소(GSQ-RCO 모델, Swift 1.5, Coder,
Unsloth file, MTP draft layer를 가져오는 원본 checkpoint)는 [ModelScope](https://www.modelscope.cn)에 같은 이름과
file로 있습니다. 설치 프로그램은 자동으로 전환하지 않습니다(다운로드 실패 시 방법 안내). `--source modelscope`
(`STRATA_SOURCE=modelscope`)를 사용하세요. ModelScope는 저장소의 현재 file을 제공하며 고정 revision은 없습니다.
따라서 각 file은 ModelScope가 공개한 SHA-256과 비교합니다(자체 증명 방식: upstream 저장소가 변경되면 자체 hash도
함께 변경). MTP tensor는 Hugging Face와 마찬가지로 고정 checkpoint 자체 hash와 비교합니다. 선택할 때 이를
안내합니다. 2026-10-05 중국 본토 PC에서 측정한 속도는 11~14 MB/s였습니다. Q2_0과 IQ3_S file이 공개 SHA-256과
일치했고 Q2_0 shard 1은 hf-mirror.com 복사본과 일치했습니다.

Linux에서는 같은 option을 `./setup.sh`에 전달합니다. `START-HERE.bat --help`에서 전체 목록을 볼 수 있습니다.
게임과 GPU 공유, MCP tool, CORS, API key와 API 등 서버 설정은 [상세 문서](DETAILS.md#using-it)에 있습니다.
