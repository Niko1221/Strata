# strata-split: the opt-in engine that splits EVERY layer of Qwen3.8-Flash-Next across 2-4 NVIDIA cards (sFNX, MIT,
# Copyright (c) 2026 John Fink, https://github.com/johnfink8/sfnx; third_party/sfnx/NOTICE.md).  Its own executable, one per card
# count and architecture, driven by serve/server.py through the same line protocol as `strata --serve`.  OFF by default
# (-DSTRATA_BUILD_SPLIT=ON), Linux + CUDA only, and nothing else in the tree depends on it.
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

if(NOT STRATA_ENABLE_CUDA OR WIN32 OR APPLE)
  message(STATUS "STRATA_BUILD_SPLIT: needs a Linux CUDA build, skipped")
  return()
endif()
if(NOT STRATA_GGML_DIR OR NOT EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-cuda/vecdotq.cuh")
  message(WARNING "STRATA_BUILD_SPLIT: -DSTRATA_GGML_DIR=<llama.cpp checkout> is needed (its ggml-cuda headers); strata-split skipped")
  return()
endif()
if(NOT STRATA_SPLIT_ARCH)
  string(REGEX MATCH "[0-9]+" STRATA_SPLIT_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
  if(NOT STRATA_SPLIT_ARCH)
    set(STRATA_SPLIT_ARCH 86)
  endif()
endif()

set(_sp ${CMAKE_CURRENT_SOURCE_DIR})
set(_sp_engine
    ${_sp}/src/gguf.cpp ${_sp}/src/model.cpp ${_sp}/src/store.cpp ${_sp}/src/moe_mma.cu ${_sp}/src/engine.cu
    ${_sp}/src/engine_common.cu ${_sp}/src/prefill.cu ${_sp}/src/session.cpp ${_sp}/src/batch.cpp ${_sp}/tools/sfnx.cpp)
set_source_files_properties(${_sp}/src/gguf.cpp ${_sp}/src/model.cpp ${_sp}/src/store.cpp ${_sp}/src/session.cpp ${_sp}/src/batch.cpp
                            ${_sp}/tools/sfnx.cpp PROPERTIES LANGUAGE CUDA)
# kernels.cu is compiled in nine parts (FNX_KPART), each with its share of the launchers, so the build runs in parallel
set(_sp_kparts "")
foreach(_p RANGE 0 8)
  set(_f ${CMAKE_CURRENT_BINARY_DIR}/kernels_p${_p}.cu)
  file(GENERATE OUTPUT ${_f} CONTENT "#define FNX_KPART ${_p}\n#include \"${_sp}/src/kernels.cu\"\n")
  list(APPEND _sp_kparts ${_f})
endforeach()

foreach(_n ${STRATA_SPLIT_CARDS})
  if(NOT _n MATCHES "^[234]$")
    message(FATAL_ERROR "STRATA_SPLIT_CARDS: 2, 3 or 4, not ${_n}")
  endif()
  set(_t strata-split-${_n})
  add_executable(${_t} ${_sp_engine} ${_sp_kparts})
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
  foreach(_tt batch_plan_test pf_ring_plan_test card_check_test rope_test sfnx_input_test)
    add_executable(split_${_tt} ${_sp}/tools/${_tt}.cpp)
    target_include_directories(split_${_tt} PRIVATE ${_sp}/src ${_sp}/tools)
    target_compile_definitions(split_${_tt} PRIVATE FNX_NCARD=${_n0} FNX_27B=0)
    target_compile_features(split_${_tt} PRIVATE cxx_std_17)
    set_target_properties(split_${_tt} PROPERTIES CUDA_ARCHITECTURES OFF)
    find_package(Threads REQUIRED)
    target_link_libraries(split_${_tt} PRIVATE Threads::Threads)
    add_test(NAME split_${_tt} COMMAND split_${_tt})
  endforeach()
  # the load-time conversion's oracle tool (tests/split/convert_check.py compares its output with gguf-py on a real file; not a ctest)
  add_executable(split_convert_dump ${_sp}/tools/convert_dump.cpp ${_sp}/src/gguf.cpp)
  target_include_directories(split_convert_dump PRIVATE ${_sp}/src ${_sp}/tools)
  target_compile_definitions(split_convert_dump PRIVATE FNX_NCARD=${_n0} FNX_27B=0)
  target_compile_features(split_convert_dump PRIVATE cxx_std_17)
  set_target_properties(split_convert_dump PROPERTIES CUDA_ARCHITECTURES OFF)
  target_link_libraries(split_convert_dump PRIVATE Threads::Threads)
endif()
