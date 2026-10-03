# strata-qwen36: Qwen3.6-35B-A3B engine.  Added to Strata's build by apply.sh (one add_subdirectory line at the end of
# Strata's CMakeLists.txt, after strata_kernels is defined).  Build for a 4090:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j --target strata-qwen36
if(NOT TARGET strata_kernels)
    message(FATAL_ERROR "strata-qwen36 needs Strata's strata_kernels target (build with CUDA enabled)")
endif()
find_package(CUDAToolkit REQUIRED)

# The batched projections run as tensor-core GEMMs through Strata's own prefill::Gemm (cuBLAS on CUDA,
# hipBLASLt/hipBLAS on HIP).  It is compiled into this target rather than linked through strata_prefill,
# which would drag in the whole qwen3.8 engine.
add_executable(strata-qwen36 main.cpp qwen36_model.cpp qwen36_kernels.cu ../prefill/gemm.cu)
set_target_properties(strata-qwen36 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 17)
target_include_directories(strata-qwen36 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include
                                                 ${_strata_gpu_include_directories})
target_link_libraries(strata-qwen36 PRIVATE strata_kernels ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
if(UNIX)
    target_link_libraries(strata-qwen36 PRIVATE pthread)
endif()
