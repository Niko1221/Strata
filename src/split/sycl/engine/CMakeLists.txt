cmake_minimum_required(VERSION 3.20)
project(strata_split_sycl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
set(STRATA_SPLIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH
    "Original all-cards split sources and runtime data")
get_filename_component(STRATA_SOURCE_DIR "${STRATA_SPLIT_SOURCE_DIR}/../.." ABSOLUTE)
if(NOT EXISTS "${STRATA_SPLIT_SOURCE_DIR}/src/model.cpp")
    message(FATAL_ERROR "Set STRATA_SPLIT_SOURCE_DIR to Strata's src/split directory")
endif()

add_library(strata_split_sycl_native OBJECT
    "${STRATA_SOURCE_DIR}/sycl/src/prefill/esimd_moe.dp.cpp"
    "${STRATA_SOURCE_DIR}/sycl/src/prefill/xmx_moe.dp.cpp"
    "${STRATA_SOURCE_DIR}/sycl/src/prefill/gemm.dp.cpp"
    "${STRATA_SOURCE_DIR}/sycl/src/kernels/cuda/iq_kernels.dp.cpp"
    "${STRATA_SOURCE_DIR}/sycl/src/kernels/cuda/qsa_prompt_attn_esimd.dp.cpp"
    "${STRATA_SOURCE_DIR}/sycl/src/kernels/cuda/qsa_select.dp.cpp"
    "${STRATA_SOURCE_DIR}/sycl/src/kernels/cuda/dequant_bf16.dp.cpp")
set_source_files_properties("${STRATA_SOURCE_DIR}/sycl/src/prefill/xmx_moe.dp.cpp"
    PROPERTIES COMPILE_DEFINITIONS STRATA_XMX_SG16)
target_include_directories(strata_split_sycl_native PRIVATE
    include "${STRATA_SOURCE_DIR}/sycl/include" "${STRATA_SOURCE_DIR}/include"
    "${STRATA_SOURCE_DIR}/third_party/ggml")
target_compile_options(strata_split_sycl_native PRIVATE
    -fsycl -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel
    -qmkl=sequential -O3 -ferror-limit=12)

add_library(strata_split_sycl_native_gdn OBJECT
    "${STRATA_SOURCE_DIR}/sycl/src/prefill/kernels.dp.cpp")
target_include_directories(strata_split_sycl_native_gdn PRIVATE
    include "${STRATA_SOURCE_DIR}/sycl/include" "${STRATA_SOURCE_DIR}/include")
target_compile_definitions(strata_split_sycl_native_gdn PRIVATE
    STRATA_PREFILL_KERNEL_NAMESPACE=fnx::native_prefill
    STRATA_PREFILL_GDN_HK=8 STRATA_PREFILL_GDN_HV=24)
target_compile_options(strata_split_sycl_native_gdn PRIVATE
    -fsycl -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel
    -O3 -ffunction-sections -fdata-sections -ferror-limit=12)

add_library(strata_split_sycl_engine OBJECT
    src/gguf.cpp src/model.cpp.dp.cpp src/store_sycl.cpp src/df2_sycl.cpp
    src/moe_mma.dp.cpp src/engine.dp.cpp src/engine_common.dp.cpp
    src/prefill.dp.cpp src/native_moe.dp.cpp src/native_gdn.dp.cpp src/native_qsa.dp.cpp
    src/session.cpp src/batch.cpp
    kernels_p0.dp.cpp kernels_p1.dp.cpp kernels_p2.dp.cpp
    kernels_p3.dp.cpp kernels_p4.dp.cpp kernels_p5.dp.cpp
    kernels_p6.dp.cpp kernels_p7.dp.cpp kernels_p8.dp.cpp)
target_include_directories(strata_split_sycl_engine PUBLIC
    include src tools vendor vendor/include vendor/ggml-cuda
    "${STRATA_SOURCE_DIR}/sycl/include" "${STRATA_SOURCE_DIR}/include")
set_source_files_properties(src/prefill.dp.cpp src/native_moe.dp.cpp PROPERTIES
    INCLUDE_DIRECTORIES "${STRATA_SOURCE_DIR}/include")
set_source_files_properties(src/native_moe.dp.cpp src/native_qsa.dp.cpp PROPERTIES COMPILE_OPTIONS "-O3")
target_compile_definitions(strata_split_sycl_engine PUBLIC
    FNX_NCARD=2 FNX_27B=0 FNX_HOSTRING=1 FNX_HIP=0 FNX_SYCL=1
    FNX_ARCH=86 FNX_DIR="${STRATA_SPLIT_SOURCE_DIR}")
target_compile_options(strata_split_sycl_engine PUBLIC
    -fsycl -fsycl-unnamed-lambda -fsycl-device-code-split=per_kernel
    -qmkl=sequential -O1 -ferror-limit=12
    -include ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl_port.hpp)
target_link_options(strata_split_sycl_engine PUBLIC
    -fsycl -fsycl-device-code-split=per_kernel -qmkl=sequential -Wl,--gc-sections)
target_link_libraries(strata_split_sycl_engine PUBLIC Threads::Threads
    strata_split_sycl_native $<TARGET_OBJECTS:strata_split_sycl_native>
    strata_split_sycl_native_gdn $<TARGET_OBJECTS:strata_split_sycl_native_gdn>)

add_executable(strata-split-sycl-2 tools/sfnx.cpp)
target_link_libraries(strata-split-sycl-2 PRIVATE strata_split_sycl_engine)
add_executable(split_sycl_input_test tools/sfnx_input_test.cpp)
target_link_libraries(split_sycl_input_test PRIVATE Threads::Threads)
target_compile_options(split_sycl_input_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME split_sycl_input_test COMMAND split_sycl_input_test)
add_executable(split_sycl_kernel_test tools/sycl_kernel_test.cpp)
target_link_libraries(split_sycl_kernel_test PRIVATE strata_split_sycl_engine)
add_executable(split_sycl_model_probe tools/sycl_model_probe.cpp)
target_link_libraries(split_sycl_model_probe PRIVATE strata_split_sycl_engine)
add_executable(split_sycl_ring_test tools/sycl_ring_test.cpp)
target_link_libraries(split_sycl_ring_test PRIVATE strata_split_sycl_engine)
add_executable(split_sycl_native_moe_test tools/sycl_native_moe_test.cpp)
target_include_directories(split_sycl_native_moe_test PRIVATE "${STRATA_SOURCE_DIR}/include")
target_link_libraries(split_sycl_native_moe_test PRIVATE strata_split_sycl_engine)
add_executable(split_sycl_native_gdn_test tools/sycl_native_gdn_test.cpp)
target_link_libraries(split_sycl_native_gdn_test PRIVATE strata_split_sycl_engine)
add_executable(split_sycl_native_attention_test tools/sycl_native_attention_test.cpp)
target_link_libraries(split_sycl_native_attention_test PRIVATE strata_split_sycl_engine)
add_executable(split_sycl_native_qsa_test tools/sycl_native_qsa_test.cpp)
target_compile_options(split_sycl_native_qsa_test PRIVATE -O3)
target_link_libraries(split_sycl_native_qsa_test PRIVATE strata_split_sycl_engine)

# Recompile prefill in its production O1 configuration, exposing its private HC
# helpers only to this executable; do not link a second copy of prefill's object.
add_executable(split_sycl_matched_kernel_bench
    tools/sycl_matched_kernel_bench.cpp src/prefill.dp.cpp
    $<FILTER:$<TARGET_OBJECTS:strata_split_sycl_engine>,EXCLUDE,prefill[.]dp[.]cpp[.]o$>
    $<TARGET_OBJECTS:strata_split_sycl_native>
    $<TARGET_OBJECTS:strata_split_sycl_native_gdn>)
target_include_directories(split_sycl_matched_kernel_bench PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,INCLUDE_DIRECTORIES>)
target_compile_definitions(split_sycl_matched_kernel_bench PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,COMPILE_DEFINITIONS>
    FNX_MATCHED_KERNEL_BENCH=1)
target_compile_options(split_sycl_matched_kernel_bench PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,COMPILE_OPTIONS>)
target_link_options(split_sycl_matched_kernel_bench PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,INTERFACE_LINK_OPTIONS>)
target_link_libraries(split_sycl_matched_kernel_bench PRIVATE Threads::Threads)

add_executable(split_sycl_prefill_failure_test EXCLUDE_FROM_ALL
    tools/sfnx.cpp src/prefill.dp.cpp
    $<FILTER:$<TARGET_OBJECTS:strata_split_sycl_engine>,EXCLUDE,prefill[.]dp[.]cpp[.]o$>
    $<TARGET_OBJECTS:strata_split_sycl_native>
    $<TARGET_OBJECTS:strata_split_sycl_native_gdn>)
target_include_directories(split_sycl_prefill_failure_test PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,INCLUDE_DIRECTORIES>)
target_compile_definitions(split_sycl_prefill_failure_test PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,COMPILE_DEFINITIONS>
    FNX_PREFILL_FAILURE_TEST=1)
target_compile_options(split_sycl_prefill_failure_test PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,COMPILE_OPTIONS>)
target_link_options(split_sycl_prefill_failure_test PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,INTERFACE_LINK_OPTIONS>)
target_link_libraries(split_sycl_prefill_failure_test PRIVATE Threads::Threads)

add_executable(split_sycl_hc_exchange_test EXCLUDE_FROM_ALL
    tools/sfnx.cpp src/prefill.dp.cpp
    $<FILTER:$<TARGET_OBJECTS:strata_split_sycl_engine>,EXCLUDE,prefill[.]dp[.]cpp[.]o$>
    $<TARGET_OBJECTS:strata_split_sycl_native>
    $<TARGET_OBJECTS:strata_split_sycl_native_gdn>)
target_include_directories(split_sycl_hc_exchange_test PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,INCLUDE_DIRECTORIES>)
target_compile_definitions(split_sycl_hc_exchange_test PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,COMPILE_DEFINITIONS>
    FNX_HC_EXCHANGE_TEST=1)
target_compile_options(split_sycl_hc_exchange_test PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,COMPILE_OPTIONS>)
target_link_options(split_sycl_hc_exchange_test PRIVATE
    $<TARGET_PROPERTY:strata_split_sycl_engine,INTERFACE_LINK_OPTIONS>)
target_link_libraries(split_sycl_hc_exchange_test PRIVATE Threads::Threads)
