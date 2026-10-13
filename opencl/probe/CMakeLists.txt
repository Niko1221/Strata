cmake_minimum_required(VERSION 3.20)
project(strata_opencl_probe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The headers and import library come from the CUDA toolkit, which is the only OpenCL SDK installed here;
# at run time the call lands in System32\OpenCL.dll, which dispatches to whatever ICDs are registered.
if(NOT CL_INCLUDE_DIR)
    set(CL_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/include"
        CACHE PATH "directory holding CL/cl.h")
endif()
if(NOT CL_LIBRARY)
    set(CL_LIBRARY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/lib/x64/OpenCL.lib"
        CACHE FILEPATH "OpenCL import library")
endif()

message(STATUS "CL_INCLUDE_DIR = ${CL_INCLUDE_DIR}")
message(STATUS "CL_LIBRARY     = ${CL_LIBRARY}")

add_executable(ocl_probe ocl_probe.cpp)
target_include_directories(ocl_probe PRIVATE "${CL_INCLUDE_DIR}")
target_link_libraries(ocl_probe PRIVATE "${CL_LIBRARY}")

add_executable(ocl_dot ocl_dot.cpp)
target_include_directories(ocl_dot PRIVATE "${CL_INCLUDE_DIR}")
target_link_libraries(ocl_dot PRIVATE "${CL_LIBRARY}")

add_executable(ocl_iq3 ocl_iq3.cpp)
target_include_directories(ocl_iq3 PRIVATE "${CL_INCLUDE_DIR}")
target_link_libraries(ocl_iq3 PRIVATE "${CL_LIBRARY}")

add_executable(ocl_iq4nl ocl_iq4nl.cpp)
target_include_directories(ocl_iq4nl PRIVATE "${CL_INCLUDE_DIR}")
target_link_libraries(ocl_iq4nl PRIVATE "${CL_LIBRARY}")

add_executable(ocl_swiglu ocl_swiglu.cpp)
target_include_directories(ocl_swiglu PRIVATE "${CL_INCLUDE_DIR}")
target_link_libraries(ocl_swiglu PRIVATE "${CL_LIBRARY}")

add_executable(ocl_iqfmt ocl_iqfmt.cpp)
target_include_directories(ocl_iqfmt PRIVATE "${CL_INCLUDE_DIR}")
target_link_libraries(ocl_iqfmt PRIVATE "${CL_LIBRARY}")

# The peer's own generated row kernel, run directly on the bytes the peer uploaded.  Used to tell a driver
# build problem apart from a wrong kernel text.
add_executable(ocl_row_repro ocl_row_repro.cpp)
target_include_directories(ocl_row_repro PRIVATE "${CL_INCLUDE_DIR}")
target_link_libraries(ocl_row_repro PRIVATE "${CL_LIBRARY}")

# The peer tier itself, against the CUDA kernels it has to reproduce.  This one needs the engine's CUDA
# kernels (native_expert_grouped / quantize_q8_1_rows), so it links the library the CUDA build produced
# rather than compiling any .cu of its own.
if(NOT STRATA_CUDA_ROOT)
    set(STRATA_CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
endif()
if(NOT STRATA_BUILD_LIB)
    set(STRATA_BUILD_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../build-cuda/strata_kernels.lib")
endif()
add_executable(ocl_peer_parity ocl_peer_parity.cpp ../../src/core/ocl_peer.cpp)
target_include_directories(ocl_peer_parity PRIVATE "${CL_INCLUDE_DIR}" "${STRATA_CUDA_ROOT}/include"
                           "../../include" "../../third_party/ggml")
target_link_libraries(ocl_peer_parity PRIVATE "${CL_LIBRARY}" "${STRATA_CUDA_ROOT}/lib/x64/cudart.lib"
                      "${STRATA_BUILD_LIB}")

# How much VRAM the MI50 will actually hold through OpenCL: the tier's single arena stops at the reported
# maximum allocation, and whether that is a wall or a hint decides if the arena gets split.
add_executable(ocl_alloc ocl_alloc.cpp)
target_include_directories(ocl_alloc PRIVATE "${CL_INCLUDE_DIR}")
target_link_libraries(ocl_alloc PRIVATE "${CL_LIBRARY}")
