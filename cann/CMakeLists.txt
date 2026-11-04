# Strata on Ascend - the CANN device layer and the first ops (P1).
#
# A separate project, like sycl/ and metal/: the CUDA engine in ../src is not compiled here yet.
# This target is the layer the port grows on: device, stream, HBM buffers, and the aclnn ops
# with a CPU oracle the parity test compares against.
#
#   cmake -S cann -B build-cann && cmake --build build-cann -j
#   ./build-cann/cann_parity
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

add_library(strata_cann STATIC src/device.cpp src/ops.cpp)
target_include_directories(strata_cann PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${ASCEND_HOME_PATH}/include"
                                              "${ASCEND_HOME_PATH}/include/aclnn"
                                              "${ASCEND_HOME_PATH}/x86_64-linux/include")
target_link_libraries(strata_cann PUBLIC ${ASCENDCL_LIB} ${NNOPBASE_LIB} ${OPAPI_LIB} ${ACL_OP_COMPILER_LIB})

add_executable(cann_parity tests/parity.cpp)
target_link_libraries(cann_parity PRIVATE strata_cann)
