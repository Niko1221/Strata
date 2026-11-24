cmake_minimum_required(VERSION 3.20)
project(strata_split_sycl_port LANGUAGES CXX)

# The experimental engine is opt-in and is never installed as strata-split-N.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Threads REQUIRED)

add_executable(split_sycl_memory_test ../tools/sycl_memory_test.cpp)
target_compile_options(split_sycl_memory_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME split_sycl_memory_test COMMAND split_sycl_memory_test)

add_library(split_sycl_runtime ../src/sycl/runtime.cpp ../src/sycl/expert_cache.cpp)
target_include_directories(split_sycl_runtime PUBLIC ../src/sycl)
target_compile_options(split_sycl_runtime PUBLIC -fsycl PRIVATE -Wall -Wextra -Werror)
target_link_options(split_sycl_runtime PUBLIC -fsycl)
target_link_libraries(split_sycl_runtime PUBLIC Threads::Threads)
add_executable(split_sycl_runtime_test ../tools/sycl_runtime_test.cpp)
target_link_libraries(split_sycl_runtime_test PRIVATE split_sycl_runtime)
target_compile_options(split_sycl_runtime_test PRIVATE -Wall -Wextra -Werror)
add_executable(split_sycl_expert_cache_test ../tools/sycl_expert_cache_test.cpp)
target_link_libraries(split_sycl_expert_cache_test PRIVATE split_sycl_runtime)
target_compile_options(split_sycl_expert_cache_test PRIVATE -Wall -Wextra -Werror)

option(STRATA_SPLIT_SYCL_ENGINE "Build the experimental two-card SYCL model engine" OFF)
if(STRATA_SPLIT_SYCL_ENGINE)
    add_subdirectory(engine)
endif()
