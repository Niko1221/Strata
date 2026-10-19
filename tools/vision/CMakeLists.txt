# tools/vision - strata-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native] [-DSTRATA_PORTABLE=OFF]
#         [-DSTRATA_VISION_HIP=ON -DSTRATA_VISION_HIP_ARCHS="gfx1200;gfx1201" + the HIP compiler the engine build uses]
#   cmake --build build-vision --target strata-vision
#
# Portable (AVX2 baseline) is the default, so an encoder built without extra flags runs on any AVX2 PC: the 0.1.32
# release's encoder was built native on an AVX-512 machine and stopped with an illegal instruction elsewhere
# (#411 #412 #419).  setup.py, which builds for the PC it runs on, passes -DSTRATA_PORTABLE=OFF.  The option is
# kept in CMakeCache.txt, so a build folder configured before this default still needs the flag (or a fresh folder).
cmake_minimum_required(VERSION 3.21)
project(strata_vision C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(STRATA_VISION_HIP "run the vision encoder on an AMD GPU (builds ggml-hip); the program is named strata-vision-hip" OFF)
set(STRATA_VISION_HIP_ARCHS "" CACHE STRING "the gfx architectures strata-vision-hip is compiled for (';'-separated)")
if(STRATA_VISION_HIP AND STRATA_VISION_CUDA)
  message(FATAL_ERROR "strata-vision: STRATA_VISION_CUDA and STRATA_VISION_HIP are two different programs")
endif()
if(STRATA_VISION_HIP AND NOT STRATA_VISION_HIP_ARCHS)
  message(FATAL_ERROR "strata-vision: STRATA_VISION_HIP needs STRATA_VISION_HIP_ARCHS, for example \"gfx1200;gfx1201\"")
endif()
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" ON)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no vcomp140.dll / libgomp needed on the user's PC
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
  # nothing above AVX2 either, even when a -D asks for it (a 9900K or a Zen 3 has neither AVX-512 nor AVX-VNNI)
  foreach(_f GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 GGML_AVX_VNNI
             GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "strata-vision: portable build (AVX2 baseline, runs on other PCs)")
else()
  # this PC's instruction set; a build folder that was portable before gets a fresh native build's settings back
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  set(GGML_OPENMP ON CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "strata-vision: native build for this CPU (STRATA_PORTABLE=OFF) - not for other PCs")
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${STRATA_VISION_CUDA} CACHE BOOL "" FORCE)
set(GGML_HIP ${STRATA_VISION_HIP} CACHE BOOL "" FORCE)
if(STRATA_VISION_HIP)
  # the engine's HIP build says the same (cmake/hip_backend.cmake); ggml-hip reads AMDGPU_TARGETS / CMAKE_HIP_ARCHITECTURES
  set(AMDGPU_TARGETS "${STRATA_VISION_HIP_ARCHS}" CACHE STRING "" FORCE)
  set(GPU_TARGETS "${STRATA_VISION_HIP_ARCHS}" CACHE STRING "" FORCE)
  set(CMAKE_HIP_ARCHITECTURES "${STRATA_VISION_HIP_ARCHS}" CACHE STRING "" FORCE)
endif()

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(strata-vision strata_vision.cpp)
target_link_libraries(strata-vision PRIVATE mtmd llama)
target_compile_features(strata-vision PRIVATE cxx_std_17)
set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(STRATA_VISION_HIP)
  # beside strata (and, in a component, the CPU reader "strata-vision"): the GPU reader is its own program, so a PC where
  # the HIP runtime does not load still has the CPU one to fall back on
  set_target_properties(strata-vision PROPERTIES OUTPUT_NAME strata-vision-hip)
  string(REPLACE ";" "," _hip_archs "${STRATA_VISION_HIP_ARCHS}")
  target_compile_definitions(strata-vision PRIVATE STRATA_VISION_HIP STRATA_VISION_HIP_ARCHS="${_hip_archs}")
endif()
if(WIN32 AND STRATA_VISION_HIP)
  # hipblas.dll (and rocblas.dll behind it) live in rocm\bin of the engine package: loaded on first use, after main() has put that folder on
  # the DLL search path, so the reader starts from any PATH and says "no GPU" itself where there is none
  target_link_libraries(strata-vision PRIVATE delayimp)
  target_link_options(strata-vision PRIVATE -Wl,/delayload:hipblas.dll)
endif()
if(UNIX AND NOT APPLE)
  # the libraries it needs are the ones in its own folder (a component folder is relocatable): RUNPATH $ORIGIN
  set_target_properties(strata-vision PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN")
  target_link_options(strata-vision PRIVATE -Wl,--enable-new-dtags)
endif()

# opens a GGUF that has tensor types llama.cpp does not know (a Gyro quantization's TQK6 / TQK7) for its vocabulary only
add_executable(strata-vision-gguf-test EXCLUDE_FROM_ALL gguf_meta_test.cpp)
target_link_libraries(strata-vision-gguf-test PRIVATE llama)
target_compile_features(strata-vision-gguf-test PRIVATE cxx_std_17)
set_target_properties(strata-vision-gguf-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# the HIP reader's architecture check (gpu_arch.h), without a GPU
add_executable(strata-vision-arch-test EXCLUDE_FROM_ALL gpu_arch_test.cpp)
target_compile_features(strata-vision-arch-test PRIVATE cxx_std_17)
set_target_properties(strata-vision-arch-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
