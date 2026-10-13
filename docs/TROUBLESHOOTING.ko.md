# 문제가 생겼나요?

[English](TROUBLESHOOTING.md) · **한국어**

자주 발생하는 문제와 해결 방법입니다. [README로 돌아가기](../README.ko.md#문제가-생겼나요).
이전 엔진의 해결 방법을 포함한 전체 오류 메시지 표는 [상세 문서](DETAILS.md#troubleshooting)에 있습니다.

## 설치 또는 시작 중

**Strata를 처음 시작할 때 PC가 멈추거나 매우 느려졌습니다.**
처음 시작할 때, 특히 최초 실행에서는 정상입니다. Strata는 RAM에 35~55 GB를 불러오고 그중 일부를 그래픽 카드용으로
고정한 뒤 GPU에 들어갈 모델 용량을 계산합니다. 마우스가 몇 분간 멈출 수 있습니다. **기다리고 창을 닫지 마세요.**
다음 시작부터는 훨씬 빠릅니다. 10분이 지나도 멈춰 있다면 PC를 다시 시작하고 다른 프로그램(브라우저는 RAM을 많이
사용함)을 닫은 뒤 다시 시도하세요. 계속 발생한다면 더 작은 크기(Q2_0 또는 IQ2_XS)를 선택하세요.

**다운로드 또는 설치 도중 중단됐습니다.**
`START-HERE.bat`(Linux: `./setup.sh`)을 다시 실행하세요. 중단된 곳부터 이어집니다.

**NVIDIA 드라이버가 너무 오래됐다고 나옵니다.**
드라이버를 업데이트하고(NVIDIA App 또는 [nvidia.com/drivers](https://www.nvidia.com/drivers)) PC를 다시 시작한 뒤
`START-HERE.bat`을 다시 실행하세요.

**8080 포트가 이미 사용 중이라고 나옵니다.**
Strata가 이미 실행 중입니다. 실행 중인 창을 찾으세요. 다른 프로그램이 포트를 사용한다면
`START-HERE.bat --port 8081`을 사용하세요. 실제로 주소가 사용 중일 때만 "already in use"라고 표시됩니다.
다른 원인(Windows가 포트를 예약했거나 설정의 `"host"`가 이 PC의 주소가 아님)은 운영체제가 반환한 메시지와
변경 방법을 함께 표시합니다(#769).

**"Windows blocked the Strata engine (or the image encoder) ... Smart App Control"이라고 나옵니다.**
새로 설치한 Windows 11에서 Smart App Control은 서명되지 않은 프로그램을 거부하며 Strata의 `strata.exe`와
`strata-vision.exe`는 서명되지 않았습니다(#735). 기능을 끄거나(Windows 보안 > 앱 및 브라우저 컨트롤 >
Smart App Control 설정. Windows를 초기화하지 않으면 다시 켤 수 없음), image encoder만 차단된다면
`START-HERE.bat --setup --vision no`를 실행하세요.

**업데이트 뒤 새 엔진이 제대로 작동하지 않습니다.**
업데이트는 교체한 엔진을 `engine\.previous`에 한 세대 보관합니다(약 210 MiB).
`python setup.py --rollback-engine`을 실행하면 이전 엔진을 복원하고 새 엔진은 그 위치에 보관합니다. 다시 실행하면
새 엔진으로 돌아갑니다(#670).

**UPDATE.bat / update.sh에서 "git pull did not succeed" 또는 "has no commit in common"이라고 나옵니다(#1276).**
2026-10-06에 저장소 이력이 정리되어 그 전에 만든 clone은 `git pull`로 업데이트할 수 없습니다. 0.1.40.2 스크립트는
tracked file이 수정되지 않았으면 이전 commit을 `pre-cleanup-backup` branch에 보관하고 clone을 새 이력으로 옮깁니다.
수정된 파일이 있으면 실행할 명령 두 개를 표시합니다. 모델, 설정과 엔진은 untracked file이므로 건드리지 않습니다.
0.1.40.2 이전 `UPDATE.bat` / `update.sh`는 이를 자동으로 처리할 수 없습니다. Strata 폴더에서 다음 명령을 한 번 실행한
뒤 평소처럼 UPDATE.bat을 사용하세요(`git status`에 수정된 파일이 있다면 먼저 `git stash` 실행).

    git fetch origin
    git branch pre-cleanup-backup
    git checkout -B main origin/main

**Python 또는 build tool을 설치할 수 없습니다.**
표시된 프로그램을 설치하고(링크가 함께 표시됨) 다시 실행하세요. 이미 완료한 작업은 유지됩니다.

**Linux: 엔진이 컴파일되지 않습니다(`unsupported GNU version` 또는 `cospi`/`sinpi`/`rsqrt`의
`exception specification is incompatible`).** 새 Linux와 CUDA toolkit 사이에 알려진 충돌 두 가지가 있습니다(#601).

- gcc 14보다 새 버전(Ubuntu 26.04 기본 버전 15): CUDA 12.x와 13.0은 이를 거부합니다. g++-14를 설치하고
  `CXX=g++-14 ./setup.sh`를 실행하세요(CC, CXX와 CUDAHOSTCXX로 build compiler 지정. 설치 프로그램이 직접 CMake에
  전달하므로 이미 설정된 build folder에도 적용됨, #1645).
- glibc 2.43과 CUDA 12.9: toolkit의 math header가 CMake 첫 test compile부터 glibc와 충돌합니다. CUDA 12.8 또는
  13.x를 사용하세요. 설치 프로그램은 발견한 가장 최신 toolkit을 선택합니다.
  `STRATA_NVCC=/usr/local/cuda-12.8/bin/nvcc ./setup.sh`로 해당 버전만 사용하게 할 수 있습니다.

**처음 시작하는 데 몇 분이 걸립니다.**
RAM으로 34~55 GB를 읽고 있습니다. 파일이 OS cache에 남아 있으면 두 번째 시작은 더 빠릅니다. Task Scheduler에서
시작하면 24배 느려질 수 있습니다. [시작 시 실행](DETAILS.md#running-it-at-startup-task-scheduler)을 참고하세요.
Linux에서는 엔진이 kernel에 모델 파일을 미리 읽도록 요청합니다(한 PC에서 cold start가 약 920초에서 70초로 단축).
`STRATA_READ_AHEAD=0`으로 끌 수 있습니다. Linux의 transparent huge page가 `always`이면 arena가 그 위에
`MADV_HUGEPAGE`를 요청하지 않습니다(조각난 시스템에서 메모리 정리에 몇 분이 걸린 사례, #771).
`STRATA_NO_ARENA_THP=1`은 어떤 설정에서도 이 요청을 생략합니다.

**설치 프로그램에서 엔진이 GitHub checksum과 일치하지 않는다고 나옵니다.**
설치 프로그램은 다운로드한 ready-made engine을 GitHub가 공개한 SHA-256과 비교합니다. 일치하지 않으면 파일이
손상됐거나 전송 중 변경된 것이므로 삭제하고 중지합니다. 다시 실행해 새로 다운로드하세요. GitHub에서 checksum을
가져올 수 없다면(오프라인, rate limit, 자체 mirror) 경고한 뒤 설치합니다. 반드시 필요한 경우에만
`STRATA_SKIP_SHA256=1`로 검사를 건너뛸 수 있습니다.

## 답변 중

**매우 느리고 디스크 표시등이 계속 깜박입니다.**
PC에 남은 RAM이 없습니다. 다른 프로그램을 닫거나 더 작은 크기(Q2_0 또는 IQ2_XS)를 선택하세요.

**답변이 "the engine stopped unexpectedly"와 함께 중단됐습니다.**
대개 RAM이 부족한 경우입니다(Linux에서는 시스템이 엔진을 중지함). 메시지를 다시 보내면 Strata가 엔진을 자동으로
시작합니다. 계속 발생하면 다른 프로그램을 닫거나 더 작은 크기를 선택하세요.

**프롬프트가 컨텍스트를 초과한다고 나옵니다.**
대화가 선택한 컨텍스트보다 깁니다. 새 대화를 시작하거나 `SETUP.bat`을 실행해 더 큰 컨텍스트를 선택하세요.

**표에 나온 속도보다 느립니다.**
그래픽 카드에 연결된 모니터와 다른 GPU 프로그램이 expert cache에서 VRAM을 사용합니다. RAM이 정격 속도보다 느리게
실행되는 경우에도 CPU 쪽 작업이 느려집니다(BIOS에서 EXPO/XMP 활성화).

**이전에 빈 내용으로 돌아온 답변이 대화 기록에서 사라졌습니다(0.1.40, #843).**
서버는 텍스트가 없는 완료된 assistant turn을 제외하므로 다음 답변에서 빈 turn을 복사하지 않습니다. 서버 환경에
`STRATA_KEEP_EMPTY_TURNS=1`을 설정하면 이전처럼 표시합니다.

**Turing 카드(RTX 20)가 약 90K 토큰보다 긴 프롬프트를 다르게 읽습니다(0.1.40, #743).**
프롬프트의 top-k 선택이 같은 id로 더 넓은 kernel을 사용합니다. `STRATA_TOPK_STREAM=0`으로 이전 방식을 복원합니다.

**`--expert-cache N`을 사용하면 `auto`보다 decode가 몇 배 느립니다(#781, #831).**
직접 지정한 N은 slot이 기록된 뒤 남은 VRAM과 비교하지 않는 byte budget입니다. 카드를 가득 채우면(log에
`0 MiB of VRAM free with everything loaded - LOW` 표시) Windows가 GPU memory를 paging하여 slot이 수백 개만 많아도
decode가 100 tok/s에서 14 tok/s로 떨어집니다. `--expert-cache auto` 또는 더 작은 N을 사용하세요.

**저 RAM PC(약 32 GB)가 `--resident-budget-gib N`에서 멈춥니다(#649).**
budget이 RAM에 고정되어 OS와 CPU worker가 사용할 공간이 부족합니다. `STRATA_RESIDENT_PIN=0`으로 paging 가능하게
하거나 더 작은 budget을 사용하세요.

**RAM 32 GB PC에서 `--resident-budget-gib N`을 사용하면 0.1.40부터 decode가 느려지고 드라이브를 많이 읽습니다
(#1194, #1116, #1085).** 시작 메시지의 "the file tier reads unbuffered"는 RAM copy 밖의 전문가를 드라이브에서
직접 읽는다는 뜻입니다. 0.1.40은 여유 RAM이 모든 전문가를 담을 수 없으면 항상 이 방식을 선택했습니다. 하지만 같은
소수가 토큰마다 반복되므로 전체의 20분의 1 크기 cache만 있어도 반복 요청 대부분을 처리할 수 있습니다. 0.1.40.2는
그만큼(4 GiB headroom 외 최소 1.5 GiB)을 요청하고, 공간이 없을 때만 cache를 통해 읽습니다. RAM 20~32 GB PC에서
드라이브 traffic이 3분의 1~6분의 1인 상태로 decode가 7~118% 빨라졌습니다. 0.1.40 / 0.1.40.1에서는
`STRATA_UNBUFFERED_LOAD=0`이 같은 동작을 하며 `=1`은 unbuffered read를 강제합니다.

**Linux: 엔진이 page-locked expert copy를 할당하는 동안 OOM으로 종료됩니다(#1250).**
Page-locked page는 driver가 한 번에 할당하며 kernel이 file cache를 반환할 때까지 기다릴 수 없습니다. 모델 파일 cache가
공간을 차지하면(실제 free RAM은 적고 "available"은 많음) host RAM이 바닥날 수 있습니다. 0.1.40.2는 copy를 page-lock하기
전에 매핑한 모델 파일 cache를 반환합니다. 실제 free RAM이 여전히 부족하면 RAM available(cgroup limit 포함)이
headroom보다 높게 유지되는 동안 1 GiB씩 가져오고, 가져온 부분만 page-lock한 뒤 나머지는 resident 상태로 두되 paging을
허용하며 경고합니다("only N of M GiB could be page-locked"). 답변은 같고 paging 가능한 부분을 GPU로 복사할 때만 CPU를
거칩니다. `STRATA_PIN_GUARD=0`으로 이 기능을 끄고, `STRATA_PIN_RESERVE_GIB=N`으로 남겨 둘 RAM을 정할 수 있습니다
(기본값: `STRATA_RESIDENT_HEADROOM_GIB`, 4 GiB).

**답변 중 Windows display driver가 재설정된 뒤 PC에서 blue screen이 발생합니다(`nvlddmkm`의 0x141 다음 0x116,
#961).** 이는 Strata가 아니라 NVIDIA driver 문제입니다. 해당 보고서에서는 같은 driver(610.88, RTX 4080 SUPER)의
다른 CUDA 프로그램에서도 같은 문제가 발생했습니다. 다른 driver(Studio 또는 이전 버전)와 낮은 power limit을
사용해 보세요. `--spec 4`에서만 발생한다면 driver가 수정될 때까지 `--spec 0`으로 verify window를 피할 수 있습니다.

**RTX 50 카드(CUDA 13.2 source build)가 의미 없는 답변을 하거나 프롬프트를 잘못 읽습니다(#892, #968).**
CUDA 13.2.0과 13.2.1의 compiler(nvcc 13.2.51)는 엔진의 일부 kernel을 sm_120용으로 잘못 컴파일합니다. RTX 5070
테스트에서 IQ2_S와 IQ3_S product가 13.2에서는 잘못됐고(relative error 0.5~1.0) 13.0에서는 정상입니다. Ready-made
engine은 13.0으로 만들어졌습니다. 직접 컴파일한다면 CUDA 13.0, 13.1 또는 13.2.2를 사용하세요(13.2.2, nvcc build
13.2.86에서 수정됨. `STRATA_NVCC=<path to its nvcc>`로 이전 버전을 최신 버전 옆에 설치해 사용 가능). 설치 프로그램은
해당 카드에서 13.2.0 또는 13.2.1을 찾으면 경고하고, 다른 13.x가 설치되어 있으면 이전 버전을 선택합니다.

**이미지가 거부되거나 느립니다.**
"this server was started without the vision encoder"는 모델이 텍스트 전용으로 설정됐다는 뜻입니다. 설치 프로그램을
다시 실행해 `--vision gpu`(또는 `--vision cpu`)를 선택하세요. 몇 초 걸리는 이미지는 encoder가 CPU에서 읽습니다
(8 core, image token 300개에서 약 3초. token이 많으면 더 오래 걸림). `--vision gpu`(NVIDIA, VRAM 약 1.4 GB)는
0.1~0.5초가 걸립니다.

## AMD 카드

**Linux에서 "No AMD GPU found (the amdgpu driver's KFD topology is empty)"라고 나옵니다.**
해당 카드에 kernel의 amdgpu driver가 load되지 않았습니다. 통합 Radeon GPU는 지원하지 않는 것으로 표시됩니다.
설치 프로그램은 발견한 모든 카드와 Strata 사용 가능 여부를 보여 줍니다.

**시작할 때 카드 이름, architecture와 build 목록을 표시한 뒤 엔진이 중지됩니다.**
다른 카드용으로 엔진을 컴파일한 상태입니다(예: Strata 폴더를 다른 PC로 옮긴 경우).
`./setup.sh --setup --backend hip`을 실행해 이 카드 architecture용 엔진을 컴파일하세요.

**Windows에서 아무 내용도 출력하기 전에 엔진이 `amdhip64_7.dll`의 `0xC0000005`로 종료됩니다(#654).**
환경의 두 가지 원인이 있을 수 있으며, 이제 서버는 엔진을 시작하기 전에 둘 다 복구하고 log에 표시합니다. 삭제된
build처럼 더 이상 존재하지 않는 ROCm 폴더를 가리키는 `HIP_PATH`, `HIP_DEVICE_LIB_PATH` 또는 `LLVM_PATH`, 그리고
엔진이 파일을 만들 수 없는 `TEMP`/`TMP` 폴더입니다(AMD runtime은 첫 kernel을 temporary file로 컴파일함).
`strata.exe`를 직접 실행한다면 `set HIP_PATH=`로 비우고 `TEMP`와 `TMP`를 `C:\Temp` 같은 일반 폴더로 지정하세요.

**Windows AMD: driver가 재설정됩니다(VIDEO_ENGINE_TIMEOUT_DETECTED / 화면 깜박임 / 답변 도중 엔진 종료).**
Windows는 GPU가 약 2초 안에 응답하지 않으면 driver를 재설정합니다(TDR). 화면이 깜박이고 엔진이 종료되며 Event
Viewer에 `VIDEO_ENGINE_TIMEOUT_DETECTED`가 기록됩니다(#613, Linux의 #579 및 #541 포함). Strata 0.1.39 이상은
log에 "the GPU stopped responding"을 표시합니다. Gfx1201 카드에서 긴 프롬프트를 읽을 때 KV streaming
(`--kv-resident 32768`, 64K 컨텍스트부터 설치 프로그램이 활성화)이 켜진 상태에서 발생했습니다. 다음을 한 번에 하나씩
순서대로 시도하고 무엇이 달라졌는지 알려 주세요.

1. **근거를 보내 주세요.** Strata 폴더의 `strata-<model>.log` 마지막 약 80줄(서버 창에도 같은 내용 표시), AMD
   driver 버전(AMD Software > System), 카드와 Windows 버전, Strata 버전, 정확한 단계(context 크기, prompt 길이,
   보낸 앱). Windows가 dump를 작성했다면 `C:\Windows\LiveKernelReports\WATCHDOG`(또는 `C:\Windows\Minidump`)의
   가장 최신 `.dmp`도 도움이 됩니다. 대화 내용은 없으며 driver state만 담깁니다.
2. **멈추는 단계 찾기:** `strata-<model>.json`의 `"env"` block에 `"STRATA_PF_STEP_SYNC": "1"`을 추가하고
   다시 시작하세요. Prompt path가 각 단계 뒤 GPU를 기다리고 250 ms보다 긴 단계를 기록하므로 log에서 문제 단계를
   확인할 수 있습니다. 더 느리므로 진단 뒤에는 제거하세요.
3. 같은 `"env"` block에 **`"STRATA_KV_HOST_DMA": "1"`** 추가: prompt 중 GPU kernel이 K/V의 RAM copy를
   쓰지 않고 DMA로 복사합니다. 답변은 같습니다.
4. **KV streaming 끄기:** 설치 프로그램을 다시 실행해 32K context를 선택하거나, VRAM에 들어가는 context 크기와
   함께 JSON의 `"args"`에서 `--kv-resident`와 그 숫자를 제거하세요.
5. **더 작은 prompt chunk:** `"args"`의 `--prefill`을 변경하세요(예: `--prefill 512`). GPU launch 한 번이 Windows의
   두 확인 사이에서 처리하는 일이 줄지만 prompt를 더 느리게 읽습니다.
6. **선택 사항이며 사용자가 결정해야 함: Windows 대기 시간 늘리기(`TdrDelay`).** 기본값은 2초입니다. Registry의
   `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`에 DWORD `TdrDelay` 값을 `10`(초)으로
   만들면 Windows가 driver를 재설정하기 전에 긴 GPU 작업이 끝날 시간을 줍니다. PC를 다시 시작해야 합니다. 증상을
   숨길 뿐 실제로 멈춘 GPU를 고치지는 않으며, GPU가 멈추면 해당 10초 동안 화면도 멈춥니다. Strata는 Windows 설정을
   바꾸지 않습니다. 되돌리려면 값을 삭제하세요.
7. **Driver:** 카드용 최신 AMD Software(Windows Update driver가 아닌 Adrenalin)를 사용하고 설치 뒤 다시 시작하세요.
   Driver 업데이트 뒤 문제가 시작됐다면 두 버전을 모두 알려 주세요. 한 버전 되돌리는 것도 좋은 테스트입니다.

**RAM이 충분한데 ROCm에서 큰 pinned host allocation이 실패합니다.**
[AMD_HIP.md](AMD_HIP.md#model-and-serving-configuration)를 참고하세요. Mapped expert mode는 전체 pinned arena를 피합니다.

**AMD(Linux)에서 "verify: timed out at layer N" 또는 "no progress for 60 s"가 나옵니다(주로
`--mmap-experts` 사용 시).** 두 보고서에서는 서버 JSON의 `env` block에 `"GPU_PINNED_MIN_XFER_SIZE": "1048576"`
(RX 6800) 또는 `"HSA_USERPTR_FOR_PAGED_MEM": "0"`(R9700 두 장)을 추가한 뒤 다시 시작해 해결했습니다. 각각 한
시스템에서만 확인된 실험적 해결책입니다. Systemd unit에서는 `MemoryHigh`가 아닌 `MemoryMax`를 사용하세요. 자세한
내용은 [AMD_HIP.md](AMD_HIP.md#linux-verify-timeouts-while-the-kernel-reclaims-host-memory-experimental-workarounds)를
참고하세요.

## 그래도 해결되지 않나요?

[전체 문제 해결 표](DETAILS.md#troubleshooting)를 확인하거나 [이슈](https://github.com/Niko1221/Strata/issues)를 열고
Strata 폴더의 `strata-<model>.log`를 첨부하세요.
