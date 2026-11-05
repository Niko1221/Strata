# Strata on Ascend - the CANN device layer and the first ops (P1), quant formats (P2).
#
# A separate project, like sycl/ and metal/: the CUDA engine in ../src is not compiled here yet.
# This target is the layer the port grows on: device, stream, HBM buffers, the aclnn ops with a
# CPU oracle, and the Q4_K/Q6_K decoders checked against ggml's own.
#
#   cmake -S cann -B build-cann && cmake --build build-cann -j
#   ./build-cann/cann_parity
#   ./build-cann/quant_parity        # needs -DSTRATA_CANN_GGML_DIR=... for the ggml oracle
#
# Requires CANN (ASCEND_HOME_PATH, default /usr/local/Ascend/cann-9.0.0).
cmake_minimum_required(VERSION 3.24)
project(strata_cann VERSION 0.1.56 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit root")
if(NOT ASCEND_HOME_PATH)
  set(ASCEND_HOME_PATH "/usr/local/Ascend/cann-9.0.0")
endif()

find_library(ASCENDCL_LIB ascendcl PATHS "${ASCEND_HOME_PATH}/lib64" "${ASCEND_HOME_PATH}/x86_64-linux/lib64"
             NO_DEFAULT_PATH REQUIRED)
find_library(NNOPBASE_LIB nnopbase PATHS "${ASCEND_HOME_PATH}/lib64" "${ASCEND_HOME_PATH}/x86_64-linux/lib64"
             NO_DEFAULT_PATH REQUIRED)
find_library(OPAPI_LIB opapi PATHS "${ASCEND_HOME_PATH}/lib64" "${ASCEND_HOME_PATH}/x86_64-linux/lib64"
             NO_DEFAULT_PATH REQUIRED)
find_library(ACL_OP_COMPILER_LIB acl_op_compiler PATHS "${ASCEND_HOME_PATH}/lib64"
             "${ASCEND_HOME_PATH}/x86_64-linux/lib64" NO_DEFAULT_PATH REQUIRED)

add_library(strata_cann STATIC src/device.cpp src/ops.cpp src/quants.cpp src/iq_quants.cpp src/gdn.cpp src/gguf.cpp src/tensor_load.cpp src/engine.cpp)
target_include_directories(strata_cann PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${ASCEND_HOME_PATH}/include"
                                              "${ASCEND_HOME_PATH}/include/aclnn"
                                              "${ASCEND_HOME_PATH}/x86_64-linux/include")
target_include_directories(strata_cann PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(strata_cann PUBLIC ${ASCENDCL_LIB} ${NNOPBASE_LIB} ${OPAPI_LIB} ${ACL_OP_COMPILER_LIB})

add_library(strata_cann_decode SHARED src/quants.cpp src/iq_quants.cpp src/decode_api.cpp)
target_include_directories(strata_cann_decode PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
                                                     "${CMAKE_CURRENT_SOURCE_DIR}/../include")

add_executable(cann_parity tests/parity.cpp)
add_executable(gdn_batched_parity tests/gdn_batched_parity.cpp)
target_link_libraries(gdn_batched_parity PRIVATE strata_cann)
add_executable(port_check tests/port_check.cpp)
target_link_libraries(port_check PRIVATE strata_cann)
add_executable(gguf_dump tests/gguf_dump.cpp)
target_link_libraries(gguf_dump PRIVATE strata_cann)
add_executable(gdn_fused_parity tests/gdn_fused_parity.cpp)
target_link_libraries(gdn_fused_parity PRIVATE strata_cann)
add_executable(topk_parity tests/topk_parity.cpp)
target_link_libraries(topk_parity PRIVATE strata_cann)
add_executable(batched_parity tests/batched_parity.cpp)
target_link_libraries(batched_parity PRIVATE strata_cann)
add_executable(bench_op tests/bench_op.cpp)
target_link_libraries(bench_op PRIVATE strata_cann)
add_executable(model_seq tests/model_seq.cpp)
target_link_libraries(model_seq PRIVATE strata_cann)
add_executable(model_forward tests/model_forward.cpp)
target_link_libraries(model_forward PRIVATE strata_cann)
add_executable(attn_cache_parity tests/attn_cache_parity.cpp)
target_link_libraries(attn_cache_parity PRIVATE strata_cann)
add_executable(attn_layer_parity tests/attn_layer_parity.cpp)
target_link_libraries(attn_layer_parity PRIVATE strata_cann)
add_executable(rope_parity tests/rope_parity.cpp)
target_link_libraries(rope_parity PRIVATE strata_cann)
add_executable(gdn_state_parity tests/gdn_state_parity.cpp)
target_link_libraries(gdn_state_parity PRIVATE strata_cann)
add_executable(gdn_block_parity tests/gdn_block_parity.cpp)
target_link_libraries(gdn_block_parity PRIVATE strata_cann)
add_executable(moe_parity tests/moe_parity.cpp)
target_link_libraries(moe_parity PRIVATE strata_cann)
add_executable(layer_parity tests/layer_parity.cpp)
target_link_libraries(layer_parity PRIVATE strata_cann)
add_executable(gguf_parity tests/gguf_parity.cpp)
target_link_libraries(gguf_parity PRIVATE strata_cann)
add_executable(gdn_parity tests/gdn_parity.cpp)
target_link_libraries(gdn_parity PRIVATE strata_cann)
target_link_libraries(cann_parity PRIVATE strata_cann)

# The ggml oracle: a llama.cpp checkout gives ggml's to_float for the quant formats. Its built
# shared library is enough; the port never ships a dependency on it.
set(STRATA_CANN_GGML_DIR "" CACHE PATH "llama.cpp checkout for the ggml dequant oracle (optional)")
set(STRATA_CANN_GGML_LIB_DIR "" CACHE PATH "directory holding libggml-base.so (optional)")

add_executable(quant_parity tests/quant_parity.cpp)
target_link_libraries(quant_parity PRIVATE strata_cann)

set(_ggml_include "${STRATA_CANN_GGML_DIR}/ggml/include")
if(NOT EXISTS "${_ggml_include}/ggml.h")
  set(_ggml_include "${STRATA_CANN_GGML_DIR}/include")
endif()
if(STRATA_CANN_GGML_DIR AND EXISTS "${_ggml_include}/ggml.h")
  set(_ggml_lib_dir "${STRATA_CANN_GGML_LIB_DIR}")
  if(NOT _ggml_lib_dir)
    set(_ggml_lib_dir "${STRATA_CANN_GGML_DIR}/build-cann2/bin")
  endif()
  find_library(GGML_BASE_LIB ggml-base PATHS "${_ggml_lib_dir}" NO_DEFAULT_PATH)
  if(GGML_BASE_LIB)
    target_compile_definitions(quant_parity PRIVATE STRATA_CANN_HAVE_GGML=1)
    target_compile_definitions(gguf_parity PRIVATE STRATA_CANN_HAVE_GGML=1)
    target_include_directories(gguf_parity PRIVATE "${_ggml_include}")
    target_link_libraries(gguf_parity PRIVATE "${GGML_BASE_LIB}")
    set_target_properties(gguf_parity PROPERTIES BUILD_RPATH "${_ggml_lib_dir}")
    target_include_directories(quant_parity PRIVATE "${_ggml_include}")
    target_link_libraries(quant_parity PRIVATE "${GGML_BASE_LIB}")
    set_target_properties(quant_parity PROPERTIES BUILD_RPATH "${_ggml_lib_dir}")
    message(STATUS "quant_parity: ggml oracle from ${GGML_BASE_LIB}")
  else()
    message(STATUS "quant_parity: libggml-base not found under ${_ggml_lib_dir}; building without the oracle")
  endif()
else()
  message(STATUS "quant_parity: no STRATA_CANN_GGML_DIR; building without the ggml oracle")
endif()
