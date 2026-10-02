# Opt-in HIP configuration. Strata's CUDA-shaped kernels target RDNA3 wave32.
# CMake/compiler discovery stays machine-independent; pass CMAKE_HIP_COMPILER when it is not on PATH.

# The RDNA3 parts Strata is built and run for. All of them are wave32 with a 64 KiB LDS budget per CU, so
# the kernels' tile sizes carry over; what differs between them is CU count and clocks, i.e. throughput.
# Adding a part means compiling for it and running the device tests on it (tests/hip, HIP_DEVICE_TESTS).
set(STRATA_HIP_SUPPORTED_ARCHS gfx1100 gfx1101 CACHE STRING "HIP architectures Strata supports")

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "Strata HIP target architecture")
endif()

# One entry per arch; a feature suffix (gfx1101:xnack-) is Strata's own spelling and is not accepted here.
foreach(_strata_hip_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  if(NOT _strata_hip_arch IN_LIST STRATA_HIP_SUPPORTED_ARCHS)
    message(FATAL_ERROR
      "Strata HIP supports '${STRATA_HIP_SUPPORTED_ARCHS}' (wave32); CMAKE_HIP_ARCHITECTURES is '${CMAKE_HIP_ARCHITECTURES}'")
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
if(UNIX AND NOT APPLE)
  # Export the interposed HIP APIs from each executable, including to BLAS DSOs.
  target_sources(strata_hip_runtime INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/hip_budget.cpp")
  target_link_libraries(strata_hip_runtime INTERFACE ${CMAKE_DL_LIBS})
  target_link_options(strata_hip_runtime INTERFACE "-Wl,--export-dynamic")
endif()
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

message(STATUS "Strata: HIP enabled, arch ${CMAKE_HIP_ARCHITECTURES}")
