# Opt-in HIP configuration. Wave32 targets: RDNA3 gfx1100 and RDNA4 gfx1200/gfx1201.
# CMake/compiler discovery stays machine-independent; pass CMAKE_HIP_COMPILER when it is not on PATH.
set(_strata_hip_archs gfx1100 gfx1200 gfx1201)
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "Strata HIP target architecture")
endif()
set(_strata_hip_req "${CMAKE_HIP_ARCHITECTURES}")
string(REPLACE " " ";" _strata_hip_req "${_strata_hip_req}")
foreach(_a IN LISTS _strata_hip_req)
  if(NOT _a IN_LIST _strata_hip_archs)
    message(FATAL_ERROR
      "Strata HIP supports wave32 gfx1100, gfx1200 and gfx1201; "
      "CMAKE_HIP_ARCHITECTURES is '${CMAKE_HIP_ARCHITECTURES}'")
  endif()
endforeach()

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
target_compile_definitions(strata_hip_runtime INTERFACE STRATA_USE_HIP=1)
target_link_libraries(strata_hip_runtime INTERFACE hip::host)
# The shim renames the CUDA runtime to HIP. MSVC host compiles take /FI; HIP and
# other host compilers take -include. The header path uses forward slashes so
# /FI does not swallow backslashes.
file(TO_CMAKE_PATH "${STRATA_HIP_COMPAT_INCLUDE_DIR}/cuda_runtime.h" _strata_hip_force)
if(MSVC)
  target_compile_options(strata_hip_runtime INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:/FI${_strata_hip_force}>")
else()
  target_compile_options(strata_hip_runtime INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:-include>"
    "$<$<COMPILE_LANGUAGE:CXX>:${_strata_hip_force}>")
endif()
target_compile_options(strata_hip_runtime INTERFACE
  "$<$<COMPILE_LANGUAGE:HIP>:-include>"
  "$<$<COMPILE_LANGUAGE:HIP>:${_strata_hip_force}>")

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

message(STATUS "Strata: HIP enabled, arch ${CMAKE_HIP_ARCHITECTURES}")
