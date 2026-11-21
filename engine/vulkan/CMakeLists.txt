if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(strata_vulkan_expert_stream LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

add_library(strata_vulkan_stream STATIC
    src/expert_stream.cpp
    src/router.cpp
    src/moe_combine.cpp
    src/bf16_gemv.cpp
    src/sampler.cpp
)
target_include_directories(strata_vulkan_stream PUBLIC
    include
    ${Vulkan_INCLUDE_DIRS}
)
target_link_libraries(strata_vulkan_stream PUBLIC Vulkan::Vulkan Threads::Threads)

# io_uring reader: one thread and one kernel wait for a whole batch of NVMe
# reads instead of one futex wakeup per read. Optional; the thread-pool reader
# remains when liburing is not installed.
find_library(STRATA_URING_LIBRARY NAMES uring)
if(STRATA_URING_LIBRARY)
    target_compile_definitions(strata_vulkan_stream PUBLIC STRATA_HAVE_LIBURING)
    target_link_libraries(strata_vulkan_stream PUBLIC ${STRATA_URING_LIBRARY})
endif()

add_executable(strata-vulkan-stream-test src/expert_stream_test.cpp)
target_link_libraries(strata-vulkan-stream-test PRIVATE strata_vulkan_stream)

add_executable(strata-vulkan-router-test src/router_test.cpp)
target_link_libraries(strata-vulkan-router-test PRIVATE strata_vulkan_stream)

add_executable(strata-vulkan-combine-test src/moe_combine_test.cpp)
target_link_libraries(strata-vulkan-combine-test PRIVATE strata_vulkan_stream)

add_executable(strata-vulkan-bf16-test src/bf16_gemv_test.cpp)
target_link_libraries(strata-vulkan-bf16-test PRIVATE strata_vulkan_stream)

add_executable(strata-vulkan-sampler-test src/sampler_test.cpp)
target_link_libraries(strata-vulkan-sampler-test PRIVATE strata_vulkan_stream)

add_executable(strata-vulkan-q8-test src/q8_0_gemv_test.cpp)
target_link_libraries(strata-vulkan-q8-test PRIVATE strata_vulkan_stream)

add_executable(strata-vulkan-iq-test src/iq_gemv_test.cpp)
target_link_libraries(strata-vulkan-iq-test PRIVATE strata_vulkan_stream)

# Compile the expert matvec shaders. The .spv files are checked in so the test
# works without glslang installed; this target regenerates them when available.
find_program(GLSLANG glslangValidator)
if(GLSLANG)
    add_custom_target(vulkan_shaders ALL
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_q4_k_gemv.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_q4_k_gemv.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_q6_k_gemv.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_q6_k_gemv.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_q8_0_gemv.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_q8_0_gemv.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_iq3_xxs_gemv.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_iq3_xxs_gemv.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_iq2_xs_gemv.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_iq2_xs_gemv.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_iq1_m_gemv.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_iq1_m_gemv.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_silu_mul.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/expert_silu_mul.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/router_gemv.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/router_gemv.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/router_topk.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/router_topk.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/moe_combine.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/moe_combine.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bf16_gemv.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bf16_gemv.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/argmax_rows.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/argmax_rows.comp.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/row_top_prob.comp
                -o ${CMAKE_CURRENT_SOURCE_DIR}/shaders/row_top_prob.comp.spv
        COMMENT "Compiling expert, router, combine, BF16, Q8 and sampler shaders -> SPIR-V"
        VERBATIM)
    add_dependencies(strata-vulkan-stream-test vulkan_shaders)
    add_dependencies(strata-vulkan-router-test vulkan_shaders)
    add_dependencies(strata-vulkan-combine-test vulkan_shaders)
    add_dependencies(strata-vulkan-bf16-test vulkan_shaders)
    add_dependencies(strata-vulkan-sampler-test vulkan_shaders)
    add_dependencies(strata-vulkan-q8-test vulkan_shaders)
    add_dependencies(strata-vulkan-iq-test vulkan_shaders)
endif()
