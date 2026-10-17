# A stand-in for hip_backend.cmake's architecture choice, configured by ../hip_arch_select_test.cmake: the real
# cmake/hip_arch_select.cmake, with find_package(hip) replaced by what ROCm's hip-config-amd.cmake does to
# GPU_TARGETS / AMDGPU_TARGETS.  No compiler, no ROCm.
#   STRATA_SOURCE_DIR  the Strata checkout
#   SIM_ROCM           5.7, 6.2 or 7.2: which ROCm's cache declarations to replay
#   SIM_CARDS          what amdgpu-arch reports for the build machine (6.2 and 7.2 run it when the targets are unset)
#   SIM_OLD_STRATA     ON: choose as Strata 0.1.43 did (it cached CMAKE_HIP_ARCHITECTURES), for the upgrade cases
# Writes "<architectures>|<source>" to arch.txt in the build folder.
cmake_minimum_required(VERSION 3.24)
project(strata_hip_arch_select_test NONE)

if(SIM_OLD_STRATA)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "Strata HIP target architecture(s), e.g. gfx1100 or gfx1100;gfx1201")
  endif()
  set(STRATA_HIP_ARCH_FROM "Strata 0.1.43")
else()
  # strata_hip_arch_select() gives each entry a probe value for one read (rule 0): it must leave every entry exactly
  # as it found it
  set(_props VALUE TYPE HELPSTRING ADVANCED STRINGS MODIFIED)
  foreach(_var IN ITEMS GPU_TARGETS AMDGPU_TARGETS)
    if(DEFINED CACHE{${_var}})
      foreach(_p IN LISTS _props)
        get_property(_before_${_var}_${_p} CACHE ${_var} PROPERTY ${_p})
      endforeach()
    endif()
  endforeach()
  include("${STRATA_SOURCE_DIR}/cmake/hip_arch_select.cmake")
  strata_hip_arch_select()
  foreach(_var IN ITEMS GPU_TARGETS AMDGPU_TARGETS)
    if(DEFINED CACHE{${_var}})
      foreach(_p IN LISTS _props)
        get_property(_after CACHE ${_var} PROPERTY ${_p})
        if(NOT "${_after}" STREQUAL "${_before_${_var}_${_p}}")
          message(FATAL_ERROR "strata_hip_arch_select() changed ${_var}'s ${_p}: '${_before_${_var}_${_p}}' -> '${_after}'")
        endif()
      endforeach()
    endif()
  endforeach()
endif()

# ---- find_package(hip) ----
set(hip_DIR "/opt/rocm/lib/cmake/hip" CACHE PATH "The directory containing a CMake configuration file for hip.")
if(SIM_ROCM STREQUAL "5.7")
  # clr rocm-5.7.1 hipamd/hip-config-amd.cmake
  set(AMDGPU_TARGETS "" CACHE STRING "AMD GPU targets to compile for")
  set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU targets to compile for")
elseif(SIM_ROCM STREQUAL "6.2")
  # clr rocm-6.2.0 hipamd/hip-config-amd.cmake
  if(NOT AMDGPU_TARGETS)
    set(AMDGPU_TARGETS "${SIM_CARDS}" CACHE STRING "AMD GPU targets to compile for")
  endif()
  set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU targets to compile for")
elseif(SIM_ROCM STREQUAL "7.2")
  # ROCm 7.2 lib/cmake/hip/hip-config-amd.cmake
  if(DEFINED AMDGPU_TARGETS AND NOT DEFINED GPU_TARGETS)
    set(GPU_TARGETS "${AMDGPU_TARGETS}")
  endif()
  if(NOT GPU_TARGETS)
    set(GPU_TARGETS "${SIM_CARDS}" CACHE STRING "AMD GPU targets to compile for")
  endif()
else()
  message(FATAL_ERROR "SIM_ROCM is '${SIM_ROCM}'; 5.7, 6.2 or 7.2")
endif()
# ----

if(NOT SIM_OLD_STRATA)
  strata_hip_arch_record()
endif()
file(WRITE "${CMAKE_BINARY_DIR}/arch.txt" "${CMAKE_HIP_ARCHITECTURES}|${STRATA_HIP_ARCH_FROM}")
