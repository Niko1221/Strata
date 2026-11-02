# strata-split: the opt-in engine that splits EVERY layer of Qwen3.8-Flash-Next across 2-4 NVIDIA or AMD cards (sFNX, MIT,
# Copyright (c) 2026 John Fink, https://github.com/johnfink8/sfnx; third_party/sfnx/NOTICE.md).  Its own executable, one per card
# count and architecture, driven by serve/server.py through the same line protocol as `strata --serve`.  OFF by default
# (-DSTRATA_BUILD_SPLIT=ON), Linux + CUDA (sm_86) or Linux + HIP (-DSTRATA_ENABLE_HIP=ON, gfx1201: R9700, RX 9070), and nothing else
# in the tree depends on it. The HIP build compiles the same sources as HIP (FNX_HIP=1): gfx12 WMMA for the expert prompt kernel
# and the attention tiles (src/hip/mma.h, moe_mma.cuh), the driver's stream memory operations through hipStreamWrite/WaitValue32
# (src/hip/cuda.h), the rest through the tree's hip_compat shim plus src/hip/fnx_hip.h.
#
# No card ever writes into another card's memory: the buffers the cards exchange sums through live in page-locked host
# memory (src/engine.cu, src/prefill.cu), so no peer access is enabled, and tests/split/test_split_no_peer.py fails the
# build's tests if a peer API call appears in this directory.
#
#   STRATA_SPLIT_CARDS   card counts to build, e.g. "2;3" (default "2": 3 and 4 cards have not been run with the host exchange)
#   STRATA_SPLIT_ARCH    the one compute capability the kernels are built for (default: the first number in CMAKE_CUDA_ARCHITECTURES, else 86);
#                        the engine refuses a card of any other architecture
set(STRATA_SPLIT_CARDS "2" CACHE STRING "card counts strata-split is built for (2, 3, 4)")
set(STRATA_SPLIT_ARCH "" CACHE STRING "compute capability strata-split is built for (e.g. 86)")

set(_fnx_hip OFF)
if(STRATA_ENABLE_HIP AND NOT STRATA_ENABLE_CUDA AND NOT WIN32 AND NOT APPLE)
  set(_fnx_hip ON)
endif()
if(NOT _fnx_hip AND (NOT STRATA_ENABLE_CUDA OR WIN32 OR APPLE))
  message(STATUS "STRATA_BUILD_SPLIT: needs a Linux CUDA or HIP build, skipped")
  return()
endif()
if(NOT STRATA_GGML_DIR OR NOT EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/vecdotq.cuh")
  message(WARNING "STRATA_BUILD_SPLIT: -DSTRATA_GGML_DIR=<llama.cpp checkout> is needed (its ggml-cuda headers); strata-split skipped")
  return()
endif()
if(_fnx_hip)
  # the kernels are built for gfx1201 only (the engine refuses any other card at start)
  if(NOT STRATA_HIP_ARCH_LIST STREQUAL "gfx1201")
    message(WARNING "STRATA_BUILD_SPLIT (HIP): the split engine is built for gfx1201 only (CMAKE_HIP_ARCHITECTURES=gfx1201), skipped")
    return()
  endif()
  set(STRATA_SPLIT_ARCH 1201)
elseif(NOT STRATA_SPLIT_ARCH)
  string(REGEX MATCH "[0-9]+" STRATA_SPLIT_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
  if(NOT STRATA_SPLIT_ARCH)
    set(STRATA_SPLIT_ARCH 86)
  endif()
endif()

set(_sp ${CMAKE_CURRENT_SOURCE_DIR})
set(_sp_engine
    ${_sp}/src/gguf.cpp ${_sp}/src/model.cpp ${_sp}/src/store.cpp ${_sp}/src/moe_mma.cu ${_sp}/src/engine.cu
    ${_sp}/src/engine_common.cu ${_sp}/src/prefill.cu ${_sp}/src/session.cpp ${_sp}/src/batch.cpp ${_sp}/tools/sfnx.cpp)
if(_fnx_hip)
  set(_fnx_lang HIP)
else()
  set(_fnx_lang CUDA)
endif()
set_source_files_properties(${_sp}/src/gguf.cpp ${_sp}/src/model.cpp ${_sp}/src/store.cpp ${_sp}/src/session.cpp ${_sp}/src/batch.cpp
                            ${_sp}/tools/sfnx.cpp PROPERTIES LANGUAGE ${_fnx_lang})
if(_fnx_hip)
  set_source_files_properties(${_sp}/src/moe_mma.cu ${_sp}/src/engine.cu ${_sp}/src/engine_common.cu ${_sp}/src/prefill.cu PROPERTIES LANGUAGE HIP)
endif()
# kernels.cu is compiled in nine parts (FNX_KPART), each with its share of the launchers, so the build runs in parallel
set(_sp_kparts "")
foreach(_p RANGE 0 8)
  set(_f ${CMAKE_CURRENT_BINARY_DIR}/kernels_p${_p}.cu)
  file(GENERATE OUTPUT ${_f} CONTENT "#define FNX_KPART ${_p}\n#include \"${_sp}/src/kernels.cu\"\n")
  if(_fnx_hip)
    set_source_files_properties(${_f} PROPERTIES LANGUAGE HIP GENERATED TRUE)
  endif()
  list(APPEND _sp_kparts ${_f})
endforeach()

foreach(_n ${STRATA_SPLIT_CARDS})
  if(NOT _n MATCHES "^[234]$")
    message(FATAL_ERROR "STRATA_SPLIT_CARDS: 2, 3 or 4, not ${_n}")
  endif()
  set(_t strata-split-${_n})
  add_executable(${_t} ${_sp_engine} ${_sp_kparts})
  if(_fnx_hip)
    # HIP: src/hip first (cuda.h, mma.h stand-ins), the force-included fnx_hip.h after the tree's cuda_runtime.h shim
    target_include_directories(${_t} PRIVATE ${_sp}/src/hip ${_sp}/src ${_sp}/tools ${STRATA_GGML_DIR}/ggml/src/ggml-cuda
                               ${STRATA_GGML_DIR}/ggml/src ${STRATA_GGML_DIR}/ggml/include)
    target_compile_definitions(${_t} PRIVATE FNX_NCARD=${_n} FNX_ARCH=${STRATA_SPLIT_ARCH} FNX_27B=0 FNX_HOSTRING=1 FNX_HIP=1 GGML_USE_HIP=1
                               FNX_DIR="${_sp}")
    target_compile_options(${_t} PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include ${_sp}/src/hip/fnx_hip.h>")
    target_link_libraries(${_t} PRIVATE strata_hip_runtime ${STRATA_HIP_BLAS_TARGETS})
    set_target_properties(${_t} PROPERTIES CXX_STANDARD 17 HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${_t} PRIVATE Threads::Threads)
    continue()
  endif()
  target_include_directories(${_t} PRIVATE ${_sp}/src ${_sp}/tools ${STRATA_GGML_DIR}/ggml/src/ggml-cuda ${STRATA_GGML_DIR}/ggml/src
                             ${STRATA_GGML_DIR}/ggml/include)
  target_compile_definitions(${_t} PRIVATE FNX_NCARD=${_n} FNX_ARCH=${STRATA_SPLIT_ARCH} FNX_27B=0 FNX_HOSTRING=1
                             FNX_DIR="${_sp}")
  target_compile_options(${_t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr -Wno-deprecated-gpu-targets>)
  set_target_properties(${_t} PROPERTIES CUDA_ARCHITECTURES "${STRATA_SPLIT_ARCH}-real" CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                        CXX_STANDARD 17)
  find_package(Threads REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${_t} PRIVATE CUDA::cublas CUDA::cuda_driver Threads::Threads)
  # libcudart shared: every runtime call the binary can make is then visible to `nm -D --undefined-only` (the no-peer test reads it)
  set_target_properties(${_t} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endforeach()

# the CPU unit tests sFNX ships (split rule, ring plan, card checks, batch plan, rope, input parser): plain C++, no GPU
if(STRATA_BUILD_TESTS)
  list(GET STRATA_SPLIT_CARDS 0 _n0)
  foreach(_tt batch_plan_test pf_ring_plan_test card_check_test rope_test sfnx_input_test peer_policy_test rccl_policy_test)
    add_executable(split_${_tt} ${_sp}/tools/${_tt}.cpp)
    target_include_directories(split_${_tt} PRIVATE ${_sp}/src ${_sp}/tools)
    target_compile_definitions(split_${_tt} PRIVATE FNX_NCARD=${_n0} FNX_27B=0)
    target_compile_features(split_${_tt} PRIVATE cxx_std_17)
    set_target_properties(split_${_tt} PROPERTIES CUDA_ARCHITECTURES OFF)
    find_package(Threads REQUIRED)
    target_link_libraries(split_${_tt} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    add_test(NAME split_${_tt} COMMAND split_${_tt})
  endforeach()
endif()
