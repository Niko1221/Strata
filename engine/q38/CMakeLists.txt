# Dense Q38 CUDA engine in the fork's dedicated area of the official engine
# tree.  Built only when STRATA_ENABLE_Q38 is on and linked into `strata`'s
# family switch (engine/src/program/generate.cpp).
set(Q38_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(Q38_SRC ${Q38_DIR}/src)
set(Q38_ENGINE ${Q38_DIR}/..)
# Q38 uses C++17 nested namespaces (`namespace strata::qwen38`) in CUDA and
# host sources; the stock engine build leaves the standard unset, so set it
# explicitly here (this is what the fork's standalone Q38 CMake did).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The official cuda.cmake creates CUDA::* imported targets inside a function
# scope, so they are not visible here; import them in this directory as well.
find_package(CUDAToolkit REQUIRED)
set(Q38_CUDA_LIBS CUDA::cudart CUDA::cublas)
if(TARGET CUDA::cublasLt)
  list(APPEND Q38_CUDA_LIBS CUDA::cublasLt)
elseif(TARGET CUDA::cublas_lt)
  list(APPEND Q38_CUDA_LIBS CUDA::cublas_lt)
endif()

# The spin8 reference profile moved with the fork data into terminal/data.
file(READ ${Q38_ENGINE}/../terminal/data/spin8/reference.profile Q38_PROFILE_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," Q38_PROFILE_BYTES "${Q38_PROFILE_HEX}")
configure_file(${Q38_SRC}/reference_profile.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/reference_profile.hpp @ONLY)

add_library(q38_spin8 STATIC
  ${Q38_SRC}/spin8/src/spin8.cpp
  ${Q38_SRC}/spin8/third_party/sha256/sha256.c)
target_include_directories(q38_spin8 PUBLIC
  ${Q38_SRC}/spin8/include ${Q38_SRC}/spin8/third_party/sha256)
target_compile_options(q38_spin8 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/std:c++17>)

add_library(strata_q38_ops STATIC
  ${Q38_SRC}/ops.cu
  ${Q38_SRC}/prefill_gemm.cu
  ${Q38_ENGINE}/src/kernels/cuda/fused_gdn.cu)
target_include_directories(strata_q38_ops PUBLIC
  ${Q38_DIR}/include ${Q38_SRC} ${Q38_ENGINE}/include)
target_compile_definitions(strata_q38_ops PUBLIC NOMINMAX)
target_link_libraries(strata_q38_ops PUBLIC ${Q38_CUDA_LIBS})
set_target_properties(strata_q38_ops PROPERTIES CUDA_ARCHITECTURES 120)
# MSVC host + nvcc: CMake's CUDA_STANDARD does not emit -std; pass it
# explicitly so C++17 nested namespaces (strata::qwen38) compile.
target_compile_options(strata_q38_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
  $<$<COMPILE_LANGUAGE:CXX>:/std:c++17>)

add_library(strata_q38_kv STATIC
  ${Q38_SRC}/kv.cu
  ${Q38_SRC}/kv_config.cpp)
target_include_directories(strata_q38_kv PRIVATE
  ${Q38_SRC}/tqcodec/include ${Q38_SRC}/tqcodec/src ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(strata_q38_kv PUBLIC strata_q38_ops q38_spin8)
set_target_properties(strata_q38_kv PROPERTIES CUDA_ARCHITECTURES 120)
target_compile_options(strata_q38_kv PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17;--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true>
  $<$<COMPILE_LANGUAGE:CXX>:/std:c++17>)

# The entry point itself: strata::qwen38::run(int,char**).
add_library(strata_q38 STATIC ${Q38_SRC}/engine.cu)
target_link_libraries(strata_q38 PUBLIC strata_q38_kv psapi)
set_target_properties(strata_q38 PROPERTIES CUDA_ARCHITECTURES 120)
target_compile_definitions(strata_q38 PUBLIC STRATA_ENABLE_Q38=1)
target_compile_options(strata_q38 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
  $<$<COMPILE_LANGUAGE:CXX>:/std:c++17>)
