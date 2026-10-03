# Intel GPU backend. The same CUDA-shaped sources are rewritten to SYCL C++
# (tools/xpu/rewrite_cuda.py) and compiled with icpx -fsycl. Do not enable this
# together with STRATA_ENABLE_CUDA or STRATA_ENABLE_HIP.

if(NOT STRATA_ENABLE_XPU)
  function(strata_xpu_adapt target)
  endfunction()
  return()
endif()

find_package(MKL CONFIG REQUIRED PATHS /opt/intel/oneapi/mkl/latest/lib/cmake/mkl)
if(NOT TARGET MKL::MKL_SYCL)
  message(FATAL_ERROR "oneMKL SYCL BLAS (MKL::MKL_SYCL) was not found. Source setvars.sh before configuring.")
endif()

set(STRATA_XPU_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/xpu_compat")
set(STRATA_XPU_REWRITE "${CMAKE_CURRENT_SOURCE_DIR}/tools/xpu/rewrite_cuda.py")

add_library(strata_xpu_runtime INTERFACE)
target_include_directories(strata_xpu_runtime BEFORE INTERFACE "${STRATA_XPU_COMPAT_INCLUDE_DIR}")
target_compile_definitions(strata_xpu_runtime INTERFACE STRATA_USE_XPU=1)
target_compile_options(strata_xpu_runtime INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-fsycl>
  "SHELL:-include ${STRATA_XPU_COMPAT_INCLUDE_DIR}/cuda_runtime.h")
target_link_options(strata_xpu_runtime INTERFACE -fsycl)
target_link_libraries(strata_xpu_runtime INTERFACE MKL::MKL_SYCL)

set(_strata_gpu_runtime_target strata_xpu_runtime)
set(_strata_gpu_blas_target strata_xpu_runtime)
set(_strata_gpu_include_directories ${STRATA_XPU_COMPAT_INCLUDE_DIR})

function(strata_xpu_adapt target)
  if(NOT STRATA_ENABLE_XPU)
    return()
  endif()
  get_target_property(_srcs ${target} SOURCES)
  if(NOT _srcs)
    return()
  endif()
  set(_new "")
  foreach(_src IN LISTS _srcs)
    if(_src MATCHES "\\.cu$")
      if(NOT IS_ABSOLUTE "${_src}")
        set(_abs "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
      else()
        set(_abs "${_src}")
      endif()
      set(_out "${CMAKE_CURRENT_BINARY_DIR}/xpu/${_src}.cpp")
      get_filename_component(_outdir "${_out}" DIRECTORY)
      file(MAKE_DIRECTORY "${_outdir}")
      execute_process(
        COMMAND python3 "${STRATA_XPU_REWRITE}" "${_abs}" "${_out}"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out_txt
        ERROR_VARIABLE _err_txt)
      if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "XPU rewrite failed for ${_src}: ${_err_txt}")
      endif()
      message(STATUS "Strata XPU: ${_out_txt}")
      list(APPEND _new "${_out}")
    else()
      list(APPEND _new "${_src}")
    endif()
  endforeach()
  set_property(TARGET ${target} PROPERTY SOURCES ${_new})
  target_link_libraries(${target} PUBLIC strata_xpu_runtime)
endfunction()

message(STATUS "Strata: Intel XPU backend enabled (SYCL / Level Zero)")
