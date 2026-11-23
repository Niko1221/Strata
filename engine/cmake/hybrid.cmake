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
  # src/split is likewise its own engine path (strata-split, STRATA_BUILD_SPLIT, OFF by default;
  # opt-in and gated by its own add_subdirectory, CMakeLists.txt). Its sources take their geometry
  # from the defines the sFNX Makefile passes (FNX_MODEL -> FNX_27B, FNX_NCARD), which
  # src/split/CMakeLists reproduces on its own targets; the hybrid glob mirrors the tree without
  # them, and the moko flavor gate broke exactly there: build/hip-src/src/split/src/gguf.hpp hit
  # `#error "FNX_27B is unset"`, model.hpp `FNX_NCARD is unset` plus the constexpr/static_assert
  # chain that reads it, and kernels.hpp joined the same way through model.hpp. The engine splits
  # every layer across 2-4 cards with host-memory exchange; a HIP copy inside strata_hip_objs has
  # no consumer in a hybrid build and no defines to compile with.
  if(NOT STRATA_BUILD_SPLIT AND _rel MATCHES "^src/split/")
    continue()
  endif()
  # src/image is not an engine path either: no engine-tree target compiles it. The
  # upstream 0.2.13+ image family (DiT, text encoder, VAE) landed after the hybrid glob
  # was written, and its mere presence in src/ made the glob copy qi21_dit.cu,
  # qwen3vl_te.cu and te_dequant.cu into strata_hip_objs - whose host-side helpers
  # (qi21_rope_table/qi21_positions in qi21_rope.cpp, te_dequant_f32_host in
  # te_dequant_host.cpp) no CUDA lane ever compiles, so the whole-archive pull made the
  # hybrid strata link fail on them. The image tools build their towers from
  # terminal/tools/ and the vision tower is its own project (tools/vision/own); keep the
  # family out of HIP reach exactly like dsv41 and split.
  if(_rel MATCHES "^src/image/")
    continue()
  endif()
  # src/inkling2 is likewise its own project - its CMakeLists opens
  # project(strata_inkling) and the engine adds the folder only behind
  # STRATA_BUILD_INKLING+STRATA_BUILD_INKLING2; neither strata nor strata-device
  # links any strata_inkling target. With v0.2.19 it gained real device lanes:
  # an NVIDIA subfolder (add_subdirectory(cuda) under STRATA_ENABLE_CUDA: the
  # Inkling backend ink_cuda_backend.cu, ink_moe_fp4_sm120.cu, ink_cuda_main.cu)
  # and its own custom-compiled HIP backend under hip/ (gated by
  # STRATA_ENABLE_HIP - which hybrid builds leave OFF). The plugin units at its
  # root (ep_dual_gpu.cu, hybrid_doorbell.cu) compile only into
  # strata_inkling_ep_cuda, CUDA-gated, and no HIP-lane object names their
  # symbols; a hip-src copy of any of them cannot even resolve the family's
  # headers, because the mirror breaks their relative includes - cuda/ units
  # hit the ink_backend.h beside the inkling2 root, and ep_dual_gpu.cu pulls
  # ink_safetensors_loader.h, whose ../image/vae/safetensors_lite.hpp sits in
  # the src/image family kept out of HIP reach just above. The whole family
  # joins the dsv41/split/image exclusions (found by the hybrid link gate on
  # momo).
  if(_rel MATCHES "^src/inkling2/")
    continue()
  endif()
  if(_rel MATCHES "\\.cu$")
    # Keep the .cu name so a sibling #include "foo.cu" still resolves, and
    # compile a .hip link of the same file. Tests, benches, and the NVFP4
    # units are not a second device path.
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
    # The same class, added by upstream v0.2.12: moe_iq4_w4a8.cu is the Turing+
    # pipelined IQ4 w4a8 kernel (mma.m16n8k32, cp.async, __cvta_generic_to_shared
    # - the moko flavor's HIP copy failed at gfx1150 on exactly those). Its one
    # caller is moe_mmq.cu's Context::run iq4_w4a8 block, which sits inside
    # `#if !defined(GGML_USE_HIP)`, and the HIP MMQ objects carry GGML_USE_HIP=1:
    # no HIP object ever names its symbols, so it joins the exclusions rather
    # than the file's own GGML_USE_HIP stub body (that body exists for the
    # gfx906 backend, which compiles every CUDA source as HIP). And
    # gdn_harness.cu is the GDN parity harness: an add_executable test target
    # whose fixture header lives in tests/gdn/ - outside the src/ mirror - so
    # it is the test class the glob already skips; upstream just did not name
    # it _test/_bench/_parity.
    if(_rel MATCHES "/mmq_nvfp4|/mmq_kq_lib\\.cu|/ggml_cuda_host\\.cu|/gdn_chunk\\.cu$|/moe_ws(_gu_iq2|_gu_iq3|_down)?\\.cu$|/gdn_harness\\.cu$|/moe_iq4_w4a8\\.cu$|_test\\.cu|_bench\\.cu|_parity\\.cu")
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
    list(APPEND _hip_files "${_dst}")
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
  "STRATA_HIP_ARCHS=\"${STRATA_HIP_ARCHS}\"")
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

set(_hip_archive "${CMAKE_BINARY_DIR}/libstrata_hip_kernels.a")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/hip-objs.txt" CONTENT
  "$<JOIN:$<TARGET_OBJECTS:strata_hip_objs>,\n>\n")
add_custom_command(
  OUTPUT "${_hip_archive}"
  COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prefix-hip.py" "${_hip_archive}" "${CMAKE_BINARY_DIR}/hip-objs.txt"
  DEPENDS strata_hip_objs "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prefix-hip.py"
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
# hip::host rides along for the same reason: hybrid_hip.cpp is IN strata_core and
# calls hipGetDeviceCount/hipLaunchKernel/..., and every target that links the
# engine now needs the HIP runtime too - the GPU-section conversation batteries
# (conversation_snapshot_test et al.) link strata_core and used to fail at ld with
# undefined hip* references. The flavor never saw it because the hybrid flavors
# build with STRATA_BUILD_TESTS=OFF; it is not flavor-only: any hybrid build with
# tests on hit it.
# Propagation is LINK-only: hip::host's imported interface carries
# INTERFACE_COMPILE_DEFINITIONS __HIP_PLATFORM_AMD__=1, and a CUDA-compiled target that
# links strata_core (strata_kernels is one) would then flip the AMD intrinsic guards in
# the shared warp-sum headers and fail nvcc. hip::amdhip64 is the runtime itself; its
# interface adds only the library, its headers as SYSTEM includes and the harmless
# USE_PROF_API.
function(_strata_hybrid_rawlibs out)
  # The imported HIP targets (hip::host and everything linking it - roc::hipblas and
  # roc::rocblas chain to hip::host) carry INTERFACE_COMPILE_DEFINITIONS
  # __HIP_PLATFORM_AMD__=1. Only the library files may enter strata_core's link
  # interface: a compile definition there would flip the AMD intrinsic guards of every
  # CUDA-compiled target in the closure.
  set(_libs "")
  foreach(_t IN ITEMS hip::amdhip64 ${_hip_math})
    if(NOT TARGET ${_t})
      continue()
    endif()
    # hip-config in ROCm 7 imports the runtime ONLY as *_RELWITHDEBINFO (the hip
    # flavor is built RelWithDebInfo); hipblas/rocblas import _RELEASE. CMake's own
    # link resolution falls back through those variants, so the resolver walks the
    # same set.
    foreach(_prop IN ITEMS IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_RELWITHDEBINFO IMPORTED_IMPLIB IMPORTED_IMPLIB_RELEASE IMPORTED_IMPLIB_RELWITHDEBINFO)
      get_target_property(_loc ${_t} ${_prop})
      if(_loc AND NOT _loc STREQUAL "${_prop}-NOTFOUND")
        list(APPEND _libs "${_loc}")
        break()
      endif()
    endforeach()
  endforeach()
  if(NOT _libs)
    message(FATAL_ERROR "hybrid: no imported HIP runtime library found")
  endif()
  set(${out} ${_libs} PARENT_SCOPE)
endfunction()
_strata_hybrid_rawlibs(_hip_link)
add_library(strata_hip_runtime INTERFACE)
target_link_libraries(strata_hip_runtime INTERFACE ${_hip_link})
target_link_libraries(strata_core PUBLIC CUDA::cublas strata_hip_runtime)

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
      strata_core strata_kernels strata_engine
      strata_prefill strata_qwen35 strata_spec
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

# The trampoline list resolves at generate time (cmake_language DEFER): every target
# the file graph creates - GPU runtime, CPU kernels, and on older trees strata_mmq -
# is then one of the wrapped hosts, whichever order the includes land in.
function(_strata_hybrid_trampolines)
  if(NOT TARGET strata)
    return()
  endif()
  set(_tramp_s "${CMAKE_BINARY_DIR}/hip-host-tramp.s")
  set(_tramp_w "${CMAKE_BINARY_DIR}/hip-host-wrap.txt")
  set(_tramp_libs "")
  set(_tramp_deps strata_hip_kernels_a)
  foreach(_lib IN ITEMS
      strata_core strata_kernels strata_engine
      strata_prefill strata_qwen35 strata_spec
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
