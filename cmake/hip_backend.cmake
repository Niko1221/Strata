# Opt-in HIP configuration. Strata's CUDA-shaped kernels target wave32 RDNA3 / RDNA4 (64 KiB LDS per workgroup,
# the signed dot4 instruction). CMake/compiler discovery stays machine-independent; pass CMAKE_HIP_COMPILER when it
# is not on PATH.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "Strata HIP target architecture(s), e.g. gfx1100 or gfx1100;gfx1201")
endif()
# Validated on real cards: gfx1100 (RX 7900 XT / XTX) and gfx1201 (RX 9070 / 9070 XT, Radeon AI PRO R9700) by the
# maintainers; gfx1101 (RX 7800 XT, #254) and gfx1200 (RX 9060 XT, #256) by their owners. gfx1102 (RX 7600) has the
# same LDS limit and dot4 instruction and passed ctest (#192), but no model run has been reported yet.
set(_strata_hip_validated gfx1100 gfx1201)
set(_strata_hip_community gfx1101 gfx1200)
set(_strata_hip_unvalidated gfx1102)
set(STRATA_HIP_ARCH_LIST "")
foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  string(REGEX REPLACE ":.*$" "" _base "${_arch}")      # gfx1100:xnack- -> gfx1100
  if(_base IN_LIST _strata_hip_validated)
  elseif(_base IN_LIST _strata_hip_community)
    message(STATUS "Strata HIP: ${_base} was validated by community reports (docs/AMD_HIP.md)")
  elseif(_base IN_LIST _strata_hip_unvalidated)
    message(WARNING "Strata HIP: ${_base} builds, but it is not validated on a real card yet; please report results")
  else()
    message(FATAL_ERROR
      "Strata HIP supports wave32 gfx1100, gfx1101, gfx1200 and gfx1201 (unvalidated: ${_strata_hip_unvalidated}); "
      "CMAKE_HIP_ARCHITECTURES is '${CMAKE_HIP_ARCHITECTURES}'")
  endif()
  list(APPEND STRATA_HIP_ARCH_LIST "${_base}")
endforeach()
list(REMOVE_DUPLICATES STRATA_HIP_ARCH_LIST)
if(NOT STRATA_HIP_ARCH_LIST)
  message(FATAL_ERROR "Strata HIP: CMAKE_HIP_ARCHITECTURES is empty")
endif()
# The compiled architectures reach the runtime device check (src/core/device.cu) as "gfx1100,gfx1201": a binary
# carried to a card it has no code for stops at startup with a clear message instead of "invalid device function".
string(REPLACE ";" "," STRATA_HIP_ARCHS "${STRATA_HIP_ARCH_LIST}")

enable_language(HIP)
find_package(hip CONFIG REQUIRED)
find_package(hipblas CONFIG REQUIRED)
find_package(hipblaslt CONFIG QUIET)

if(NOT TARGET hip::host)
  message(FATAL_ERROR "The ROCm hip CMake package did not provide hip::host")
endif()
if(NOT TARGET roc::hipblas)
  message(FATAL_ERROR "The ROCm hipblas CMake package did not provide roc::hipblas")
endif()
if(TARGET roc::hipblaslt)
  set(STRATA_HIPBLASLT_AVAILABLE ON)
else()
  set(STRATA_HIPBLASLT_AVAILABLE OFF)
  message(STATUS "Strata: hipBLASLt not found; solution-table dispatch is unavailable")
endif()

# HIP's link step produces a PIE; make Strata and ggml objects PIC for the ROCm linker.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(STRATA_HIP_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/hip_compat")
add_library(strata_hip_runtime INTERFACE)
target_include_directories(strata_hip_runtime BEFORE INTERFACE
  "${STRATA_HIP_COMPAT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(strata_hip_runtime INTERFACE STRATA_USE_HIP=1 "STRATA_HIP_ARCHS=\"${STRATA_HIP_ARCHS}\"")
target_link_libraries(strata_hip_runtime INTERFACE hip::host)
foreach(_language IN ITEMS CXX HIP)
  target_compile_options(strata_hip_runtime INTERFACE
    "$<$<COMPILE_LANGUAGE:${_language}>:-include>"
    "$<$<COMPILE_LANGUAGE:${_language}>:${STRATA_HIP_COMPAT_INCLUDE_DIR}/cuda_runtime.h>")
endforeach()

# CMake does not infer HIP from Strata's existing CUDA-shaped .cu suffixes.
file(GLOB_RECURSE _strata_hip_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")
if(_strata_hip_sources)
  set_source_files_properties(${_strata_hip_sources} PROPERTIES LANGUAGE HIP)
endif()
foreach(_source IN ITEMS tests/hip/intrinsics.cpp tests/hip/native_qsa_score.cpp)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    set_source_files_properties("${_source}" PROPERTIES LANGUAGE HIP)
  endif()
endforeach()

message(STATUS "Strata: HIP enabled, arch ${STRATA_HIP_ARCHS}")
