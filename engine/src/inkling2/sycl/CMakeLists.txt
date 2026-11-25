# engine/src/inkling2/sycl/CMakeLists.txt - Intel Arc SYCL (dg2 + bmg_g31 B70) backend plugin for strata-inkling
# CRITICAL: Zero reqd_sub_group_size(8); strictly SG16!
# MIT License - Copyright (c) 2026 Strata contributors

set(_ink_eng_root "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

add_library(strata_inkling_sycl STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/ink_sycl_backend.cpp"
)
target_include_directories(strata_inkling_sycl PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${_ink_eng_root}/include"
  "${_ink_eng_root}/src/hy4flash/sycl"
  "${_ink_eng_root}/third_party/stb"
)

set(STRATA_SYCL_TARGETS "spir64,intel_gpu_acm_g10,intel_gpu_bmg_g21,intel_gpu_bmg_g31"
    CACHE STRING "SYCL AOT/JIT targets for Intel Arc Alchemist + Battlemage")

if(MSVC OR WIN32)
  target_compile_options(strata_inkling_sycl PRIVATE /fsycl /O2 -fno-fast-math "-fsycl-targets=${STRATA_SYCL_TARGETS}")
  target_link_options(strata-inkling PRIVATE /fsycl "-fsycl-targets=${STRATA_SYCL_TARGETS}")
else()
  target_compile_options(strata_inkling_sycl PRIVATE -fsycl -O3 -fno-fast-math "-fsycl-targets=${STRATA_SYCL_TARGETS}")
  target_link_options(strata-inkling PRIVATE -fsycl "-fsycl-targets=${STRATA_SYCL_TARGETS}")
endif()

# Optional Intel oneMKL SYCL BLAS (unlocks Intel Arc XMX DPAS GEMM on Alchemist & Battlemage!)
set(_ink_mkl_root "$ENV{MKLROOT}")
if(NOT _ink_mkl_root AND EXISTS "/opt/intel/oneapi/mkl/latest/include/oneapi/mkl/blas.hpp")
  set(_ink_mkl_root "/opt/intel/oneapi/mkl/latest")
endif()
if(_ink_mkl_root AND EXISTS "${_ink_mkl_root}/include/oneapi/mkl/blas.hpp")
  message(STATUS "Inkling SYCL: enabling Intel oneMKL XMX GEMM (${_ink_mkl_root})")
  target_compile_definitions(strata_inkling_sycl PRIVATE INK_SYCL_USE_ONEMKL=1)
  target_include_directories(strata_inkling_sycl PRIVATE "${_ink_mkl_root}/include")
  if(NOT WIN32)
    target_link_options(strata-inkling PRIVATE "-L${_ink_mkl_root}/lib" "-Wl,-rpath,${_ink_mkl_root}/lib")
    target_link_libraries(strata_inkling_sycl PUBLIC mkl_sycl_blas mkl_intel_lp64 mkl_sequential mkl_core)
  endif()
endif()

target_link_libraries(strata_inkling_sycl PUBLIC strata_inkling_core strata_zstd_dec)
target_compile_definitions(strata_inkling_core PUBLIC INKLING_ENABLE_SYCL=1)
target_link_libraries(strata-inkling PRIVATE strata_inkling_sycl)

# Explicitly built GPU checks. An optional SQ-20 artifact exercises the real expert codecs.
if(STRATA_BUILD_TESTS)
  add_executable(strata_inkling_sycl_kernel_test tests/bmg_kernels.cpp)
  set(STRATA_INKLING_TEST_MODEL "" CACHE FILEPATH "Optional local SQ-20 model for the Battlemage kernel test")
  add_test(NAME inkling_bmg_kernels COMMAND strata_inkling_sycl_kernel_test ${STRATA_INKLING_TEST_MODEL})
  set_tests_properties(inkling_bmg_kernels PROPERTIES SKIP_RETURN_CODE 77 LABELS "sycl;battlemage")
else()
  add_executable(strata_inkling_sycl_kernel_test EXCLUDE_FROM_ALL tests/bmg_kernels.cpp)
endif()
target_link_libraries(strata_inkling_sycl_kernel_test PRIVATE strata_inkling_sycl strata_inkling_core)
target_include_directories(strata_inkling_sycl_kernel_test PRIVATE
  "$<TARGET_PROPERTY:strata_inkling_sycl,INCLUDE_DIRECTORIES>")
target_compile_definitions(strata_inkling_sycl_kernel_test PRIVATE
  "$<TARGET_PROPERTY:strata_inkling_sycl,COMPILE_DEFINITIONS>")
target_compile_options(strata_inkling_sycl_kernel_test PRIVATE "$<TARGET_PROPERTY:strata_inkling_sycl,COMPILE_OPTIONS>")
target_link_options(strata_inkling_sycl_kernel_test PRIVATE "$<TARGET_PROPERTY:strata-inkling,LINK_OPTIONS>")
