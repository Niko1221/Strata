# strata-qwen36: Qwen3.6-35B-A3B engine.  Added to Strata's build by apply.sh (one add_subdirectory line at the end of
# Strata's CMakeLists.txt, after strata_kernels is defined).  Build for a 4090:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j --target strata-qwen36
if(NOT TARGET strata_kernels)
    message(FATAL_ERROR "strata-qwen36 needs Strata's strata_kernels target (build with CUDA enabled)")
endif()
find_package(CUDAToolkit REQUIRED)

add_executable(strata-qwen36 main.cpp qwen36_model.cpp qwen36_kernels.cu)
set_target_properties(strata-qwen36 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 17)
target_include_directories(strata-qwen36 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(strata-qwen36 PRIVATE strata_kernels CUDA::cudart)
if(UNIX)
    target_link_libraries(strata-qwen36 PRIVATE pthread)
endif()
