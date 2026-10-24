cmake_minimum_required(VERSION 3.16)
project(strata_ascend_s2_probe LANGUAGES CXX)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Standalone experiment. The engine's build and default runtime are unchanged.
set(SOC_VERSION "Ascend910B3" CACHE STRING "Ascend SoC used for this experiment")
set(RUN_MODE "npu" CACHE STRING "Ascend C execution mode")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Installed CANN SDK")
set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake"
    CACHE PATH "Directory containing ascendc.cmake")
get_filename_component(STRATA_DEFAULT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(STRATA_SOURCE_ROOT "${STRATA_DEFAULT_ROOT}" CACHE PATH "Strata source for the existing CPU reference")
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

ascendc_library(strata_s2 SHARED dequant_s2.cpp)
target_link_libraries(strata_s2 PRIVATE ascendcl)
add_library(strata_s2_reference SHARED reference.cpp)
target_include_directories(strata_s2_reference PRIVATE "${STRATA_SOURCE_ROOT}/include")
set_target_properties(strata_s2_reference PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
