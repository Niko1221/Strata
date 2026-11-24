include(${CMAKE_CURRENT_LIST_DIR}/hy4_flash_mmq.cmake)
target_sources(strata_kernels PRIVATE src/kernels/cuda/q4_pair_reuse.cu)
set_source_files_properties(src/kernels/cuda/q4_pair_reuse.cu src/hy4flash_gguf/ihc_fused.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math")
add_library(strata_hy4_flash_cpu STATIC
    src/hy4flash_gguf/host_memory.cpp src/hy4flash_gguf/cpu/expert_layout.cpp
    src/hy4flash_gguf/cpu/native_expert.cpp src/hy4flash_gguf/cpu/native_q81.cpp src/hy4flash_gguf/cpu/pool.cpp)
if(STRATA_CPU_X86)
  target_sources(strata_hy4_flash_cpu PRIVATE src/hy4flash_gguf/cpu/native_q81_avx2.cpp)
  if(MSVC)
    set_source_files_properties(src/hy4flash_gguf/cpu/native_q81_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(src/hy4flash_gguf/cpu/native_q81_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
  endif()
endif()
target_link_libraries(strata_hy4_flash_cpu PUBLIC strata_kernels_cpu PRIVATE ggml-cpu)
add_library(strata_hy4_flash STATIC src/hy4flash_gguf/cpu_experts.cpp src/hy4flash_gguf/engine.cpp src/hy4flash_gguf/geometry.cpp src/hy4flash_gguf/ihc.cu src/hy4flash_gguf/ihc_fused.cu
                                  src/hy4flash_gguf/engram.cpp src/hy4flash_gguf/attention.cu
                                  src/hy4flash_gguf/engram_layer.cu src/hy4flash_gguf/mat.cu src/hy4flash_gguf/head.cu
                                  src/hy4flash_gguf/engram_io.cpp
                                  src/hy4flash_gguf/forward.cu src/hy4flash_gguf/kv_cache.cu
                                  src/hy4flash_gguf/prefill_moe.cu src/hy4flash_gguf/indexer.cu
                                  src/hy4flash_gguf/sampler.cpp src/hy4flash_gguf/conversation_cache.cpp src/hy4flash_gguf/serve.cpp src/hy4flash_gguf/mtp.cu
                                  src/hy4flash_gguf/dfly.cu src/hy4flash_gguf/dfly_loop.cpp src/hy4flash_gguf/dfly_policy.cpp)
target_include_directories(strata_hy4_flash PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           PRIVATE ${_strata_gpu_include_directories})
target_link_libraries(strata_hy4_flash PUBLIC strata_engine PRIVATE strata_hy4_flash_cpu strata_hy4_flash_mmq strata_prefill strata_warnings
                      ${_strata_gpu_runtime_target})
target_link_libraries(strata PRIVATE strata_hy4_flash)
target_compile_definitions(strata PRIVATE STRATA_HAVE_HY4_FLASH=1)
add_executable(hy4_flash_verify_rows EXCLUDE_FROM_ALL ../tools/hy4_flash/verify_rows.cpp)
target_link_libraries(hy4_flash_verify_rows PRIVATE strata_hy4_flash)
if(NOT STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  target_sources(strata_hy4_flash PRIVATE src/hy4flash_gguf/vision_gguf.cu)
  target_link_libraries(strata_hy4_flash PRIVATE CUDA::cublas)
  target_compile_definitions(strata_hy4_flash PRIVATE STRATA_HY4_FLASH_VISION=1)
  target_sources(strata_hy4_flash PRIVATE src/hy4flash_gguf/gpu_sampler.cu)
  target_compile_definitions(strata_hy4_flash PUBLIC STRATA_HY4_FLASH_GPU_SAMPLER=1)
  add_executable(hy4_flash_layer_major_output_test EXCLUDE_FROM_ALL tests/hy4flash_gguf/layer_major_output_test.cpp)
  target_link_libraries(hy4_flash_layer_major_output_test PRIVATE strata_hy4_flash CUDA::cudart)
  add_executable(hy4_flash_kv_q8_test EXCLUDE_FROM_ALL tests/hy4flash_gguf/kv_q8_test.cu)
  target_link_libraries(hy4_flash_kv_q8_test PRIVATE strata_hy4_flash CUDA::cudart)
  add_executable(hy4_flash_gpu_sampler_test EXCLUDE_FROM_ALL tests/hy4flash_gguf/gpu_sampler_test.cpp)
  target_link_libraries(hy4_flash_gpu_sampler_test PRIVATE strata_hy4_flash)
endif()
if(NOT STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  add_executable(hy4_flash_window_pair_test EXCLUDE_FROM_ALL tests/hy4flash_gguf/window_pair_test.cu)
  target_link_libraries(hy4_flash_window_pair_test PRIVATE strata_kernels CUDA::cudart)
endif()
if(STRATA_BUILD_TESTS)
  add_executable(router_sigmoid_parity src/kernels/router_sigmoid_parity.cpp)
  target_include_directories(router_sigmoid_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(router_sigmoid_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME router_sigmoid_parity COMMAND router_sigmoid_parity)
  add_executable(hy4_flash_sampler_test tests/hy4flash_gguf/sampler_test.cpp src/hy4flash_gguf/sampler.cpp)
  target_include_directories(hy4_flash_sampler_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME hy4_flash_sampler_test COMMAND hy4_flash_sampler_test)
  add_executable(hy4_flash_expert_resources_test tests/hy4flash_gguf/expert_resources_test.cpp)
  target_link_libraries(hy4_flash_expert_resources_test PRIVATE strata_hy4_flash_cpu)
  add_test(NAME hy4_flash_expert_resources_test COMMAND hy4_flash_expert_resources_test)
  add_executable(hy4_flash_cpu_split_fit_test tests/hy4flash_gguf/cpu_split_fit_test.cpp)
  target_include_directories(hy4_flash_cpu_split_fit_test PRIVATE include src/hy4flash_gguf)
  add_test(NAME hy4_flash_cpu_split_fit_test COMMAND hy4_flash_cpu_split_fit_test)
  # the confidence-gated Dfly window: its choices, censored labels and costs (no model, no GPU)
  add_executable(hy4_flash_dfly_policy_test tests/hy4flash_gguf/dfly_policy_test.cpp src/hy4flash_gguf/dfly_policy.cpp)
  target_include_directories(hy4_flash_dfly_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME hy4_flash_dfly_policy_test COMMAND hy4_flash_dfly_policy_test)
  # the model's shape from its metadata, on synthetic files (no model, no GPU); --real SHARD reads a real one
  add_executable(hy4_flash_geometry_test tests/hy4flash_gguf/geometry_test.cpp src/hy4flash_gguf/geometry.cpp)
  target_link_libraries(hy4_flash_geometry_test PRIVATE strata_artifact)
  add_test(NAME hy4_flash_geometry_test COMMAND hy4_flash_geometry_test)
  # index.txt v3 / v4: the columns a wrong version would misread (no model, no GPU)
  add_executable(hy4_flash_pack_index_test tests/hy4flash_gguf/pack_index_test.cpp)
  target_include_directories(hy4_flash_pack_index_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(hy4_flash_pack_index_test PRIVATE strata_core strata_artifact
                        ${_strata_gpu_runtime_target})
  add_test(NAME hy4_flash_pack_index_test COMMAND hy4_flash_pack_index_test)
  if(STRATA_NATIVE_EXPERTS)
    add_executable(hy4_flash_cpu_experts_test tests/hy4flash_gguf/cpu_experts_test.cpp
                   src/hy4flash_gguf/cpu_experts.cpp)
    target_link_libraries(hy4_flash_cpu_experts_test PRIVATE strata_hy4_flash_cpu ggml-base)
    add_test(NAME hy4_flash_cpu_experts_test COMMAND hy4_flash_cpu_experts_test)
    add_executable(hy4_flash_q81_vector_quant_test tests/hy4flash_gguf/q81_vector_quant_test.cpp)
    target_link_libraries(hy4_flash_q81_vector_quant_test PRIVATE strata_hy4_flash_cpu ggml-base)
    add_test(NAME hy4_flash_q81_vector_quant_test COMMAND hy4_flash_q81_vector_quant_test)
    set_tests_properties(hy4_flash_q81_vector_quant_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  add_executable(hy4_flash_engram_hash_test tests/hy4flash_gguf/engram_hash_test.cpp src/hy4flash_gguf/engram.cpp)
  target_include_directories(hy4_flash_engram_hash_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME hy4_flash_engram_hash_test COMMAND hy4_flash_engram_hash_test)

  add_executable(hy4_flash_rope_test tests/hy4flash_gguf/rope_test.cu src/hy4flash_gguf/attention.cu src/hy4flash_gguf/mat.cu src/hy4flash_gguf/prefill_moe.cu)
  target_include_directories(hy4_flash_rope_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                             ${_strata_gpu_include_directories})
  # strata_kernels because attention.cu also holds mla_project, which reuses native_mmvq and
  # rms_norm_weighted - the rope kernel alone would not need it.
  target_link_libraries(hy4_flash_rope_test PRIVATE strata_hy4_flash_mmq strata_prefill strata_kernels strata_warnings
                        ${_strata_gpu_runtime_target})
  add_test(NAME hy4_flash_rope_test COMMAND hy4_flash_rope_test)

  add_executable(hy4_flash_attention_window_parity tests/hy4flash_gguf/attention_window_parity.cu
                 src/hy4flash_gguf/attention.cu src/hy4flash_gguf/mat.cu src/hy4flash_gguf/prefill_moe.cu)
  target_include_directories(hy4_flash_attention_window_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(hy4_flash_attention_window_parity PRIVATE strata_hy4_flash_mmq strata_artifact strata_prefill
                       strata_kernels strata_warnings ${_strata_gpu_runtime_target})
  add_test(NAME hy4_flash_attention_window_parity COMMAND hy4_flash_attention_window_parity)
  set_tests_properties(hy4_flash_attention_window_parity PROPERTIES SKIP_RETURN_CODE 77)

  # Compare contiguous and scattered expert slots using identical synthetic weights.
  add_executable(hy4_flash_prefill_slot_parity tests/hy4flash_gguf/prefill_slot_parity.cu
                 src/hy4flash_gguf/prefill_moe.cu)
  target_include_directories(hy4_flash_prefill_slot_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                             ${_strata_gpu_include_directories})
  target_link_libraries(hy4_flash_prefill_slot_parity PRIVATE strata_hy4_flash_mmq
                        strata_warnings ${_strata_gpu_runtime_target})
  add_test(NAME hy4_flash_prefill_slot_parity COMMAND hy4_flash_prefill_slot_parity)

  add_executable(hy4_flash_expert_window_parity tests/hy4flash_gguf/expert_window_parity.cpp)
  target_include_directories(hy4_flash_expert_window_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(hy4_flash_expert_window_parity PRIVATE strata_kernels strata_artifact ${_strata_gpu_runtime_target})
  add_test(NAME hy4_flash_expert_window_parity COMMAND hy4_flash_expert_window_parity)
  set_tests_properties(hy4_flash_expert_window_parity PROPERTIES SKIP_RETURN_CODE 77)

  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(hy4_flash_q4_pair_reuse tests/hy4flash_gguf/q4_pair_reuse.cu)
    target_include_directories(hy4_flash_q4_pair_reuse PRIVATE ${_strata_gpu_include_directories})
    target_compile_options(hy4_flash_q4_pair_reuse PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    target_link_libraries(hy4_flash_q4_pair_reuse PRIVATE strata_kernels strata_artifact CUDA::cudart)
    add_test(NAME hy4_flash_q4_pair_reuse COMMAND hy4_flash_q4_pair_reuse)
    add_test(NAME hy4_flash_q4_pair_reuse_pdl COMMAND hy4_flash_q4_pair_reuse)
    set_tests_properties(hy4_flash_q4_pair_reuse PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "STRATA_DF_PDL=0;STRATA_MMVQ_ROWS=-1;STRATA_Q4_PAIR_REUSE=1")
    set_tests_properties(hy4_flash_q4_pair_reuse_pdl PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "STRATA_DF_PDL=1;STRATA_MMVQ_ROWS=-1;STRATA_Q4_PAIR_REUSE=1")
  endif()

  # The Dfly drafter's load-time checks. Needs the drafter and the target (STRATA_DFLY_GGUF, STRATA_HY4_FLASH_GGUF)
  # and a device; without them it exits 77 and ctest reports it SKIPPED, not passed.
  add_executable(hy4_flash_dfly_load tests/hy4flash_gguf/dfly_load_test.cpp)
  target_include_directories(hy4_flash_dfly_load PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                             ${_strata_gpu_include_directories})
  target_link_libraries(hy4_flash_dfly_load PRIVATE strata_hy4_flash strata_artifact ${_strata_gpu_runtime_target})
  add_test(NAME hy4_flash_dfly_load COMMAND hy4_flash_dfly_load)
  set_tests_properties(hy4_flash_dfly_load PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(hy4_flash_ihc_parity tests/hy4flash_gguf/ihc_parity.cu src/hy4flash_gguf/ihc.cu src/hy4flash_gguf/ihc_fused.cu
                 src/hy4flash_gguf/mat.cu src/hy4flash_gguf/attention.cu src/hy4flash_gguf/prefill_moe.cu)
  target_include_directories(hy4_flash_ihc_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(hy4_flash_ihc_parity PRIVATE strata_hy4_flash_mmq strata_prefill strata_kernels strata_warnings
                        ${_strata_gpu_runtime_target})
  add_test(NAME hy4_flash_ihc_parity COMMAND hy4_flash_ihc_parity)
  # its hc_pre reference is the unfused path, so the fusion switch must not leak in from the caller
  set_tests_properties(hy4_flash_ihc_parity PROPERTIES ENVIRONMENT "STRATA_HY4_FLASH_HC_FUSE=0")
endif()
