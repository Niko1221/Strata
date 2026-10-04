cmake_minimum_required(VERSION 3.18)
project(dsv4_strata_ext LANGUAGES CXX)
enable_testing()

# Eredita o imposta lo standard C++ (Strata usa C++20, manteniamo la coerenza)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Backend device: di default quello EMULATO su CPU (src/gpu.cpp), che non richiede CUDA.
# Con -DDSV4_WITH_CUDA=ON si compila src/gpu_cuda.cu AL POSTO di src/gpu.cpp: i due backend
# definiscono gli stessi simboli dsv4::gpu, quindi non possono mai stare nello stesso binario.
option(DSV4_WITH_CUDA "Build the real CUDA device layer (src/gpu_cuda.cu) instead of the CPU-emulated one" OFF)

if(DSV4_WITH_CUDA)
    # sm_89 = RTX 4090 di questa macchina. Per un'altra scheda: -DCMAKE_CUDA_ARCHITECTURES="80;89".
    # Va impostato PRIMA di enable_language(CUDA): dopo, cmake ha gia' messo il suo default (52) nella
    # cache e "NOT DEFINED" non lo vede piu', quindi il kernel verrebbe compilato per la scheda sbagliata.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
    # nvcc 12.8 e' in /usr/local/cuda-12.8: se cmake non lo trova da solo, diglielo.
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        find_program(DSV4_NVCC NAMES nvcc nvcc-12.8 PATHS /usr/local/cuda/bin /usr/local/cuda-12.8/bin
                                                /usr/local/cuda-12/bin ENV PATH)
        if(DSV4_NVCC)
            set(CMAKE_CUDA_COMPILER "${DSV4_NVCC}")
        endif()
    endif()
    enable_language(CUDA)
    set(DSV4_GPU_SRC src/gpu_cuda.cu)
else()
    set(DSV4_GPU_SRC src/gpu.cpp)
endif()

# OpenMP: senza, i "#pragma omp parallel for" di cpu_mv/gpu::matvec/experts_hit sono inerti e il MoE
# gira su un core solo. Obbligatorio se trovato, con avviso esplicito altrimenti.
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found: dsv4_run will run SINGLE-THREADED (roughly 20x slower).")
endif()

# 1. Definiamo la Libreria Core Additiva
# src/iq_tables.cpp e' generato: python3 tools/extract_iq_tables.py
add_library(dsv4_core
    src/gguf_header.cpp
    src/config.cpp
    src/mem_plan.cpp
    src/ops.cpp
    src/dequant.cpp
    src/iq_tables.cpp
    ${DSV4_GPU_SRC}
    src/sampler.cpp
    src/serve_loop.cpp
    src/model.cpp
)

# Riferiamo i percorsi d'inclusione interni al modulo
target_include_directories(dsv4_core PUBLIC include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(dsv4_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Consente a dsv4_core di accedere agli header di Strata qualora servissero in Phase 2
target_include_directories(dsv4_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# 2. Definiamo l'eseguibile principale dsv4_run
add_executable(dsv4_run src/dsv4_run_main.cpp)
target_link_libraries(dsv4_run PRIVATE dsv4_core)   # + strata_core in Phase 2 (read-only use)

# 2b. Il planner di residenza degli esperti (CPU only)
add_executable(dsv4_plan src/dsv4_plan_main.cpp)
target_link_libraries(dsv4_plan PRIVATE dsv4_core)

# 3. Definiamo i Test interni del modulo
add_executable(test_dsv4_core tests/test_dsv4_core.cpp)
target_link_libraries(test_dsv4_core PRIVATE dsv4_core)
add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE dsv4_core)
# La matematica di decode dei kernel CUDA (include/dsv4/dq_traits.hpp), verificata su CPU contro
# dequant_row()/row_dot(): nessun touch della GPU, nessun nvcc.
add_executable(test_dq_traits tests/test_dq_traits.cpp)
target_link_libraries(test_dq_traits PRIVATE dsv4_core)
add_test(NAME core COMMAND test_dsv4_core)
add_test(NAME ops COMMAND test_ops ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden.txt)
add_test(NAME dq_traits COMMAND test_dq_traits)
# -DDSV4_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89   (usa src/gpu_cuda.cu al posto di src/gpu.cpp)
# -DSTRATA_ROOT=/path/to/Strata   (Strata e' solo LETTO: include paths + link, mai modificato)
