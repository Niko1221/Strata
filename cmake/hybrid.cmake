# STRATA_HYBRID. The HIP copies use a hybrid ordinal. hipSetDevice takes a HIP index.
# include/strata/hip_compat/hybrid_ordinal.h translates at that boundary.
if(NOT STRATA_HYBRID)
  return()
endif()

# Targets created after this file is included (strata_mmq holds moe_fused.cu)
# pick this up. The marker has to be in that CUDA object and in the HIP object.
add_compile_definitions(STRATA_HYBRID=1)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  message(FATAL_ERROR "STRATA_HYBRID needs CMAKE_HIP_ARCHITECTURES")
endif()
set(CMAKE_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "" FORCE)
enable_language(HIP)
# A configure that stops while HIP is enabling caches CMAKE_HIP_FLAGS_<CONFIG>
# empty, and the next configure of that folder then builds at -O0. Same repair
# as cmake/hip_backend.cmake.
foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(STRIP "${CMAKE_HIP_FLAGS_${_cfg}_INIT}" _strata_init)
  if("${CMAKE_HIP_FLAGS_${_cfg}}" STREQUAL "" AND NOT "${_strata_init}" STREQUAL "")
    set(CMAKE_HIP_FLAGS_${_cfg} "${_strata_init}" CACHE STRING
        "Flags used by the HIP compiler during ${_cfg} builds." FORCE)
  endif()
endforeach()
enable_language(ASM)
find_package(hip CONFIG REQUIRED)
find_package(hipblas CONFIG REQUIRED)
find_package(rocblas CONFIG REQUIRED)
# Same optional dispatcher as cmake/hip_backend.cmake. The HIP copy of gemm.cu
# is what runs on a HIP device. The CUDA prefill library is nvcc, so the
# __HIPCC__ blocks there are compiled out.
find_package(hipblaslt CONFIG QUIET)
if(TARGET roc::hipblaslt)
  set(STRATA_HIPBLASLT_AVAILABLE ON)
  message(STATUS "Strata hybrid: hipBLASLt solution-table dispatch is available")
else()
  set(STRATA_HIPBLASLT_AVAILABLE OFF)
  message(STATUS "Strata hybrid: hipBLASLt not found; solution-table dispatch is unavailable")
endif()
set(_hip_math roc::hipblas roc::rocblas)
if(STRATA_HIPBLASLT_AVAILABLE)
  list(APPEND _hip_math roc::hipblaslt)
endif()

set(_compat "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/hip_compat/cuda_runtime.h")
# Mirror src/ so a quoted include of a sibling header resolves next to the .hip
# file. .cu files are linked as .hip; everything else keeps its name.
file(GLOB_RECURSE _src_files CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
set(_hip_files "")
set(_mmq_files "")
foreach(_f IN LISTS _src_files)
  if(IS_DIRECTORY "${_f}")
    continue()
  endif()
  file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_f}")
  set(_compile FALSE)
  # src/dsv41 is its own engine path (strata-dsv41, STRATA_BUILD_DSV41, OFF by default). The CUDA
  # lane never compiles it into strata, so its HIP copies must not be globbed into strata_hip_objs
  # either - their host-side helpers are not linked there and the engine link would fail.
  if(NOT STRATA_BUILD_DSV41 AND _rel MATCHES "^src/dsv41/")
    continue()
  endif()
  if(_rel MATCHES "\\.cu$")
    # Keep the .cu name so a sibling #include "foo.cu" still resolves, and
    # compile a .hip link of the same file. Tests, benches, and the NVFP4
    # units are not a second device path. moe_mmq.cu is compiled below
    # without the CUDA compat header: it includes ggml's HIP headers.
    set(_cu_dst "${CMAKE_BINARY_DIR}/hip-src/${_rel}")
    get_filename_component(_cu_dir "${_cu_dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${_cu_dir}")
    if(NOT EXISTS "${_cu_dst}")
      file(CREATE_LINK "${_f}" "${_cu_dst}" SYMBOLIC)
    endif()
    # gdn_chunk.cu is the Volta recurrence. The CUDA prefill target compiles it.
    # Its file-scope __half is behind a non-HIP include, and the HIP call
    # sites are inside STRATA_EXPERIMENTAL_SM60, which this build leaves off.
    # The moe_ws family (upstream #3752, landed with the 0.2.7 rebase) is the
    # GP100 weight-stationary experts: pure PTX (prmt/mad.wide/f16x2 asm), a
    # strata_mmq source only under STRATA_EXPERIMENTAL_SM60, gated at run time
    # to cc-6.0-only machines. A HIP copy of it cannot compile (PTX constraints)
    # and could never dispatch on an AMD device, so it stays out of the glob too.
    if(_rel MATCHES "/mmq_nvfp4|/mmq_kq_lib\\.cu|/ggml_cuda_host\\.cu|/gdn_chunk\\.cu$|/moe_ws(_gu_iq2|_gu_iq3|_down)?\\.cu$|_test\\.cu|_bench\\.cu|_parity\\.cu")
      continue()
    endif()
    string(REGEX REPLACE "\\.cu$" ".hip" _rel_out "${_rel}")
    set(_compile TRUE)
  else()
    set(_rel_out "${_rel}")
  endif()
  set(_dst "${CMAKE_BINARY_DIR}/hip-src/${_rel_out}")
  get_filename_component(_dir "${_dst}" DIRECTORY)
  file(MAKE_DIRECTORY "${_dir}")
  if(NOT EXISTS "${_dst}")
    file(CREATE_LINK "${_f}" "${_dst}" SYMBOLIC)
  endif()
  if(_compile)
    if(_dst MATCHES "/moe_mmq\\.hip$")
      list(APPEND _mmq_files "${_dst}")
    else()
      list(APPEND _hip_files "${_dst}")
    endif()
  endif()
endforeach()
if(NOT _hip_files)
  message(FATAL_ERROR "STRATA_HYBRID: no .cu sources")
endif()

add_library(strata_hip_objs OBJECT ${_hip_files})
set_source_files_properties(${_hip_files} PROPERTIES LANGUAGE HIP)
target_include_directories(strata_hip_objs BEFORE PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/hip_compat"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml")
if(STRATA_GGML_DIR)
  target_include_directories(strata_hip_objs PRIVATE
    "${STRATA_GGML_DIR}/ggml/include"
    "${STRATA_GGML_DIR}/ggml/src"
    "${STRATA_GGML_DIR}/ggml/src/ggml-cuda")
endif()
# CMAKE_HIP_ARCHITECTURES is a CMake list (semicolons). The runtime check splits
# STRATA_HIP_ARCHS on commas, the same way cmake/hip_backend.cmake does.
string(REPLACE " " ";" _hip_arch_norm "${CMAKE_HIP_ARCHITECTURES}")
set(STRATA_HIP_ARCH_LIST "")
# Same allow-list as cmake/hip_backend.cmake.
set(_strata_hip_validated gfx1100 gfx1201)
set(_strata_hip_community gfx1101 gfx1200)
set(_strata_hip_unvalidated gfx1010 gfx1011 gfx1012 gfx1102 gfx1103 gfx1030 gfx1031 gfx1032 gfx1034 gfx1150 gfx1151 gfx1152)
foreach(_arch IN LISTS _hip_arch_norm)
  if(_arch STREQUAL "")
    continue()
  endif()
  string(REGEX REPLACE ":.*$" "" _base "${_arch}")
  if(_base STREQUAL "gfx906")
    message(FATAL_ERROR "STRATA_HYBRID does not build gfx906 (wave64). STRATA_HIP_GFX906 is a separate backend.")
  elseif(_base IN_LIST _strata_hip_validated)
  elseif(_base IN_LIST _strata_hip_community)
    message(STATUS "Strata hybrid: ${_base} was validated by community reports (docs/AMD_HIP.md)")
  elseif(_base IN_LIST _strata_hip_unvalidated)
    message(WARNING "Strata hybrid: ${_base} builds, but it is not validated on a real card yet; please report results")
  else()
    message(FATAL_ERROR
      "Strata hybrid supports the same wave32 architectures as the HIP backend "
      "(unvalidated: ${_strata_hip_unvalidated}); CMAKE_HIP_ARCHITECTURES is '${CMAKE_HIP_ARCHITECTURES}'")
  endif()
  list(APPEND STRATA_HIP_ARCH_LIST "${_base}")
endforeach()
list(REMOVE_DUPLICATES STRATA_HIP_ARCH_LIST)
if(NOT STRATA_HIP_ARCH_LIST)
  message(FATAL_ERROR "STRATA_HYBRID: CMAKE_HIP_ARCHITECTURES is empty")
endif()
string(REPLACE ";" "," STRATA_HIP_ARCHS "${STRATA_HIP_ARCH_LIST}")
# Same names as gfx_arch_is_gfx11_wmma. A gfx11 prefix is wider than the kernels.
set(_wmma_arches gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151)
set(_hip_wmma OFF)
foreach(_arch IN LISTS STRATA_HIP_ARCH_LIST)
  if(_arch IN_LIST _wmma_arches)
    set(_hip_wmma ON)
  endif()
endforeach()
set(_hip_defs
  STRATA_USE_HIP=1
  STRATA_HYBRID=1
  "STRATA_HIP_ARCHS=\"${STRATA_HIP_ARCHS}\""
  STRATA_PREFILL_MMQ=1)
if(STRATA_Q6K_EXPERTS)
  list(APPEND _hip_defs STRATA_Q6K_EXPERTS=1)
endif()
if(_hip_wmma)
  list(APPEND _hip_defs STRATA_WMMA_GFX11=1)
endif()
# rocBLAS is required above. hipBLASLt is optional. Both dispatchers stay
# idle until STRATA_ROCBLAS_TUNING or STRATA_HIPBLASLT_TUNING names a table.
list(APPEND _hip_defs STRATA_ROCBLAS_AVAILABLE=1)
if(STRATA_HIPBLASLT_AVAILABLE)
  list(APPEND _hip_defs STRATA_HIPBLASLT_AVAILABLE=1)
endif()
target_compile_definitions(strata_hip_objs PRIVATE ${_hip_defs})
# CMake drops a repeated -include. SHELL keeps each header on the command.
target_compile_options(strata_hip_objs PRIVATE
  "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include ${_compat}>"
  "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/hip_compat/hybrid_ordinal.h>"
  "$<$<COMPILE_LANGUAGE:HIP>:-fPIC>")
foreach(_hip IN LISTS _hip_files)
  if(_hip MATCHES "elementwise\\.hip$|quantize_act\\.hip$")
    set_source_files_properties("${_hip}" PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()
endforeach()
set_target_properties(strata_hip_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(strata_hip_objs PRIVATE hip::host ${_hip_math})

# Gemm::native_mmq (HIP only) calls mmq::f16_to_f32 and mmq::set_bounds, which
# moe_mmq.cu defines under __HIPCC__. ggml's own HIP headers supply the rest
# of that file, so this unit does not take the CUDA compat -include.
if(NOT _mmq_files)
  message(FATAL_ERROR "STRATA_HYBRID: src/prefill/moe_mmq.cu missing")
endif()
if(NOT STRATA_GGML_DIR)
  message(FATAL_ERROR "STRATA_HYBRID: moe_mmq needs STRATA_GGML_DIR")
endif()
# Same ggml MMQ units the CUDA strata_mmq target compiles. Wrappers keep
# LANGUAGE HIP off the store .cu files, which that target still compiles.
set(_mmq_types q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
# The CUDA side builds these in strata_kq. A pure HIP build puts them in the
# MMQ library. The AMD device needs the same instances for STRATA_DENSE_MMQ.
if(STRATA_MMQ_KQUANTS)
  list(APPEND _mmq_types q4_k q5_k q5_1 q6_k)
endif()
if(STRATA_ORCA_Q4KS_MMQ)
  list(APPEND _mmq_types q5_0)
endif()
set(_mmq_wrap "${CMAKE_BINARY_DIR}/hip-src/ggml-cuda")
file(MAKE_DIRECTORY "${_mmq_wrap}/template-instances")
set(_qwrap "${_mmq_wrap}/quantize.hip")
file(WRITE "${_qwrap}" "#include \"quantize.cu\"\n")
list(APPEND _mmq_files "${_qwrap}")
foreach(_t IN LISTS _mmq_types)
  set(_iwrap "${_mmq_wrap}/template-instances/mmq-instance-${_t}.hip")
  file(WRITE "${_iwrap}" "#include \"template-instances/mmq-instance-${_t}.cu\"\n")
  list(APPEND _mmq_files "${_iwrap}")
endforeach()
# The CUDA copy of ggml_cuda_info lists NVIDIA devices only. A HIP launch
# that indexes it with a hybrid ordinal reads warp_size 0. This copy uses
# hipGetDevice and the AMD arch name. prefix-hip sends the HIP MMQ calls
# here, and the CUDA table stays NVIDIA-only. The loop above already links
# the .cu beside this wrapper and leaves it out of the compat-header objects.
set(_hwrap "${CMAKE_BINARY_DIR}/hip-src/src/prefill/ggml_cuda_host.hip")
file(WRITE "${_hwrap}" "#include \"ggml_cuda_host.cu\"\n")
list(APPEND _mmq_files "${_hwrap}")
add_library(strata_hip_mmq OBJECT ${_mmq_files})
set_source_files_properties(${_mmq_files} PROPERTIES LANGUAGE HIP)
target_include_directories(strata_hip_mmq PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${STRATA_GGML_DIR}/ggml/include"
  "${STRATA_GGML_DIR}/ggml/src"
  "${STRATA_GGML_DIR}/ggml/src/ggml-cuda")
target_compile_definitions(strata_hip_mmq PRIVATE
  STRATA_USE_HIP=1
  STRATA_HYBRID=1
  GGML_USE_HIP=1
  "STRATA_HIP_ARCHS=\"${STRATA_HIP_ARCHS}\""
  STRATA_PREFILL_MMQ=1)
if(STRATA_Q6K_EXPERTS)
  target_compile_definitions(strata_hip_mmq PRIVATE STRATA_Q6K_EXPERTS=1)
endif()
if(STRATA_MMQ_KQUANTS)
  target_compile_definitions(strata_hip_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
endif()
if(STRATA_ORCA_Q4KS_MMQ)
  target_compile_definitions(strata_hip_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
endif()
target_compile_options(strata_hip_mmq PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:-fPIC>")
set_target_properties(strata_hip_mmq PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(strata_hip_mmq PRIVATE hip::host ${_hip_math})

set(_hip_archive "${CMAKE_BINARY_DIR}/libstrata_hip_kernels.a")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/hip-objs.txt" CONTENT
  "$<JOIN:$<TARGET_OBJECTS:strata_hip_objs>,\n>\n$<JOIN:$<TARGET_OBJECTS:strata_hip_mmq>,\n>\n")
add_custom_command(
  OUTPUT "${_hip_archive}"
  COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prefix-hip.py" "${_hip_archive}" "${CMAKE_BINARY_DIR}/hip-objs.txt"
  DEPENDS strata_hip_objs strata_hip_mmq "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prefix-hip.py"
  VERBATIM)
add_custom_target(strata_hip_kernels_a DEPENDS "${_hip_archive}")

set(_wraps
  cudaDeviceCanAccessPeer cudaDeviceDisablePeerAccess cudaDeviceEnablePeerAccess
  cudaDeviceGetAttribute cudaDeviceGetStreamPriorityRange cudaDeviceSynchronize
  cudaDriverGetVersion
  cudaEventCreate cudaEventCreateWithFlags cudaEventDestroy cudaEventElapsedTime
  cudaEventQuery cudaEventRecord cudaEventSynchronize
  cudaFree cudaFreeHost
  cudaFuncGetAttributes cudaFuncGetName cudaFuncSetAttribute
  cudaGetDevice cudaGetDeviceCount cudaGetDeviceProperties_v2
  cudaGetDriverEntryPointByVersion cudaGetErrorString cudaGetLastError cudaGetSymbolAddress
  cudaGraphDestroy cudaGraphExecDestroy cudaGraphGetEdges_v2 cudaGraphGetNodes
  cudaGraphInstantiate cudaGraphKernelNodeGetParams cudaGraphLaunch cudaGraphNodeGetType
  cudaGraphUpload
  cudaHostAlloc cudaHostGetDevicePointer cudaHostRegister cudaHostUnregister
  cudaInitDevice cudaLaunchHostFunc cudaLaunchKernel cudaLaunchKernelExC
  cudaMalloc cudaMallocHost
  cudaMemcpy cudaMemcpy2DAsync cudaMemcpyAsync cudaMemcpyPeer cudaMemcpyPeerAsync
  cudaMemGetInfo cudaMemset cudaMemsetAsync
  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags cudaOccupancyMaxActiveClusters
  cudaPeekAtLastError
  __cudaPopCallConfiguration __cudaPushCallConfiguration
  cudaRuntimeGetVersion cudaSetDevice
  cudaStreamBeginCapture cudaStreamCreate cudaStreamCreateWithFlags cudaStreamCreateWithPriority
  cudaStreamDestroy cudaStreamEndCapture cudaStreamGetCaptureInfo_v3 cudaStreamIsCapturing
  cudaStreamQuery cudaStreamSynchronize cudaStreamWaitEvent
  cublasCreate_v2 cublasDestroy_v2 cublasGemmEx cublasGemmStridedBatchedEx
  cublasSetMathMode cublasSetStream_v2 cublasSetWorkspace_v2 cublasSgemm_v2)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
  # CUDA 13 exports only the plain cudaGetDeviceProperties, cudaGraphGetEdges,
  # and cudaStreamGetCaptureInfo, with the signatures CUDA 12 kept on the
  # _v2/_v3 names. The wrap name must match the name the link references,
  # and the shim pairs in src/core/hybrid_runtime.cpp flip on the same
  # switch. cmake 4 no longer knows the bare list(REPLACE), so the swap
  # uses the TRANSFORM form with anchored patterns.
  list(TRANSFORM _wraps REPLACE "^cudaGetDeviceProperties_v2$" "cudaGetDeviceProperties")
  list(TRANSFORM _wraps REPLACE "^cudaGraphGetEdges_v2$" "cudaGraphGetEdges")
  list(TRANSFORM _wraps REPLACE "^cudaStreamGetCaptureInfo_v3$" "cudaStreamGetCaptureInfo")
endif()
set(_wrap_flags "-Wl,--export-dynamic")
foreach(_w IN LISTS _wraps)
  list(APPEND _wrap_flags "-Wl,--wrap=${_w}")
endforeach()

target_compile_definitions(strata_core PUBLIC STRATA_HYBRID=1 "STRATA_HIP_ARCHS=\"${STRATA_HIP_ARCHS}\"")
target_sources(strata_core PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/core/hybrid_runtime.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/core/hybrid_hip.cpp")
target_link_options(strata_core INTERFACE ${_wrap_flags})
# The shims call __real_cublas*. strata-device links strata_core and not
# strata_prefill, so cuBLAS has to travel with the shims.
target_link_libraries(strata_core PUBLIC CUDA::cublas)

foreach(_tgt IN ITEMS strata strata-device)
  if(TARGET ${_tgt})
    target_link_libraries(${_tgt} PRIVATE hip::host ${_hip_math})
  endif()
endforeach()

# The kernel archive is for the engine. strata-device only enumerates.
# Whole-archive is appended after the static libs, and it introduces
# references to host .cpp symbols those libs define. Rescan them.
if(TARGET strata)
  add_dependencies(strata strata_hip_kernels_a)
  set(_rescan "")
  foreach(_lib IN ITEMS
      strata_core strata_kernels strata_kernels_cpu strata_engine
      strata_prefill strata_qwen35 strata_spec strata_mmq strata_mmq_w4a4
      ggml-base ggml-cpu)
    if(TARGET ${_lib})
      list(APPEND _rescan "$<TARGET_FILE:${_lib}>")
    endif()
  endforeach()
  target_sources(strata PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/program/hip_probe.cpp")
  target_compile_definitions(strata PRIVATE STRATA_HYBRID=1)
  target_link_libraries(strata PRIVATE
    -Wl,--whole-archive "${_hip_archive}" -Wl,--no-whole-archive
    "-Wl,--start-group" ${_rescan} "-Wl,--end-group")
endif()

# strata_mmq is created later in this file. The trampoline list has to include
# it, or the fused-expert host is never one of the wrapped symbols.
function(_strata_hybrid_trampolines)
  if(NOT TARGET strata)
    return()
  endif()
  set(_tramp_s "${CMAKE_BINARY_DIR}/hip-host-tramp.s")
  set(_tramp_w "${CMAKE_BINARY_DIR}/hip-host-wrap.txt")
  set(_tramp_libs "")
  set(_tramp_deps strata_hip_kernels_a)
  foreach(_lib IN ITEMS
      strata_core strata_kernels strata_kernels_cpu strata_engine
      strata_prefill strata_qwen35 strata_spec strata_mmq strata_mmq_w4a4
      ggml-base ggml-cpu)
    if(TARGET ${_lib})
      target_compile_definitions(${_lib} PRIVATE STRATA_HYBRID=1)
      list(APPEND _tramp_libs "$<TARGET_FILE:${_lib}>")
      list(APPEND _tramp_deps ${_lib})
    endif()
  endforeach()
  add_custom_command(
    OUTPUT "${_tramp_s}" "${_tramp_w}"
    COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip-host-tramp.py"
            "${_tramp_s}" "${_tramp_w}" "${_hip_archive}" ${_tramp_libs}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip-host-tramp.py" "${_hip_archive}" ${_tramp_deps}
    VERBATIM)
  add_library(strata_hip_tramp STATIC "${_tramp_s}")
  set_source_files_properties("${_tramp_s}" PROPERTIES GENERATED TRUE)
  set_target_properties(strata_hip_tramp PROPERTIES LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON)
  add_dependencies(strata strata_hip_tramp)
  # The trampoline archive and the CUDA archives are one rescan group. --wrap
  # sends the engine's calls to the trampoline, and the trampoline's __real_
  # references have to pull the nvcc objects back out of those archives.
  target_link_libraries(strata PRIVATE
    "-Wl,--start-group" "$<TARGET_FILE:strata_hip_tramp>" ${_tramp_libs} "-Wl,--end-group")
  target_link_options(strata PRIVATE "LINKER:@${_tramp_w}")
endfunction()
cmake_language(DEFER CALL _strata_hybrid_trampolines)
