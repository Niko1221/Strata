# Standalone Vulkan probe for the Strata Vulkan backend feasibility study.
# Mirrors how sycl/ is a separate project: nothing here touches the main build.
cmake_minimum_required(VERSION 3.24)
project(strata-vulkan-probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Vulkan REQUIRED)

# gemv.comp -> gemv.spv next to the executables (glslc from the SDK)
set(GLSLC "$ENV{VULKAN_SDK}/Bin/glslc.exe")
if(NOT EXISTS "${GLSLC}")
  set(GLSLC "C:/VulkanSDK/1.4.357.0/Bin/glslc.exe")
endif()
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/gemv.spv
  COMMAND ${GLSLC} --target-env=vulkan1.1 ${CMAKE_CURRENT_SOURCE_DIR}/gemv.comp -o ${CMAKE_BINARY_DIR}/gemv.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gemv.comp
  COMMENT "compiling gemv.comp -> gemv.spv")
add_custom_target(spv ALL DEPENDS ${CMAKE_BINARY_DIR}/gemv.spv)

add_executable(vk_probe device_probe.cpp)
target_link_libraries(vk_probe PRIVATE Vulkan::Vulkan)

add_executable(vk_gemv gemv_probe.cpp)
target_link_libraries(vk_gemv PRIVATE Vulkan::Vulkan)
target_compile_definitions(vk_gemv PRIVATE STRATA_GEMV_SPV="${CMAKE_BINARY_DIR}/gemv.spv")
add_dependencies(vk_gemv spv)

# iq3_gu.comp -> iq3_gu.spv: the IQ3_XXS expert row, checked against the engine's integer path
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/iq3_gu.spv
  COMMAND ${GLSLC} --target-env=vulkan1.1 ${CMAKE_CURRENT_SOURCE_DIR}/iq3_gu.comp -o ${CMAKE_BINARY_DIR}/iq3_gu.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/iq3_gu.comp
  COMMENT "compiling iq3_gu.comp -> iq3_gu.spv")
add_custom_target(spv_iq3 ALL DEPENDS ${CMAKE_BINARY_DIR}/iq3_gu.spv)

add_executable(vk_iq3 iq3_probe.cpp)
target_link_libraries(vk_iq3 PRIVATE Vulkan::Vulkan)
target_compile_definitions(vk_iq3 PRIVATE STRATA_IQ3_SPV="${CMAKE_BINARY_DIR}/iq3_gu.spv")
add_dependencies(vk_iq3 spv_iq3)

# iq3_fast.comp -> one module per (lanes per row, rows per lane, dot, chunk bound).  The bench
# builds the file name at run time, so adding an axis here needs no change in the C++.
#   lanes_per_row : 32 (an NVIDIA wave) or 64 (the gfx906 wavefront)
#   rows_per_lane : how many rows share one activation read
#   dot           : scalar expansion, or dotPacked4x8EXT when glslc can emit it
#   chunk bound   : push constant, or compile-time for n_embd = 2560 (80 chunks)
set(_probe_src "${CMAKE_BINARY_DIR}/dotprobe.comp")
file(WRITE "${_probe_src}"
"#version 450\n#extension GL_EXT_integer_dot_product : require\nlayout(local_size_x=1) in;\nlayout(set=0,binding=0) readonly buffer A { int a[]; };\nlayout(set=0,binding=1) writeonly buffer B { int b[]; };\nvoid main(){ b[0] = dotPacked4x8EXT(a[0], a[1], a[2]); }\n")
execute_process(COMMAND "${GLSLC}" --target-env=vulkan1.3 "${_probe_src}" -o "${CMAKE_BINARY_DIR}/dotprobe.spv"
                RESULT_VARIABLE _probe_rc OUTPUT_QUIET ERROR_QUIET)
if(_probe_rc EQUAL 0)
  set(STRATA_GLSLC_HAS_DOT ON)
  message(STATUS "glslc: dotPacked4x8EXT available, building the packed-dot kernels")
else()
  set(STRATA_GLSLC_HAS_DOT OFF)
  message(STATUS "glslc: dotPacked4x8EXT missing, falling back to the scalar dot kernels")
endif()

set(_fast_spv)
foreach(_wg "32" "64")
  foreach(_r "1" "2" "4" "8")
    foreach(_shape "dyn" "n80")
      foreach(_dot "scalar" "dp4a")
        if(_dot STREQUAL "dp4a" AND NOT STRATA_GLSLC_HAS_DOT)
          continue()
        endif()
        set(_out "${CMAKE_BINARY_DIR}/iq3_w${_wg}_r${_r}_${_dot}_${_shape}.spv")
        set(_defs -DWG=${_wg} -DR=${_r})
        if(_shape STREQUAL "n80")
          list(APPEND _defs -DNCHUNK=80)
        endif()
        if(_dot STREQUAL "dp4a")
          list(APPEND _defs -DVK_DP4A=1)
        endif()
        add_custom_command(
          OUTPUT ${_out}
          COMMAND ${GLSLC} --target-env=vulkan1.3 ${_defs} ${CMAKE_CURRENT_SOURCE_DIR}/iq3_fast.comp -o ${_out}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/iq3_fast.comp
          COMMENT "glsl iq3 ${_wg}x${_r} ${_dot} ${_shape}")
        list(APPEND _fast_spv ${_out})
      endforeach()
    endforeach()
  endforeach()
endforeach()
add_custom_target(spv_iq3_fast ALL DEPENDS ${_fast_spv})

add_executable(vk_iqbench iq_bench.cpp)
target_link_libraries(vk_iqbench PRIVATE Vulkan::Vulkan)
target_compile_definitions(vk_iqbench PRIVATE
  STRATA_SPV_DIR="${CMAKE_BINARY_DIR}"
  STRATA_IQBENCH_BIN="${CMAKE_CURRENT_SOURCE_DIR}/iq_bench.bin")
if(STRATA_GLSLC_HAS_DOT)
  target_compile_definitions(vk_iqbench PRIVATE STRATA_GLSLC_HAS_DOT=1)
endif()
add_dependencies(vk_iqbench spv_iq3_fast)

# iq4nl.comp -> the down projection, one module per (lanes per row, rows per lane)
set(_iq4_spv)
foreach(_wg "32" "64")
  foreach(_r "1" "4")
    set(_out "${CMAKE_BINARY_DIR}/iq4nl_w${_wg}_r${_r}.spv")
    add_custom_command(
      OUTPUT ${_out}
      COMMAND ${GLSLC} --target-env=vulkan1.3 -DWG=${_wg} -DR=${_r} ${CMAKE_CURRENT_SOURCE_DIR}/iq4nl.comp -o ${_out}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/iq4nl.comp
      COMMENT "glsl iq4nl ${_wg}x${_r}")
    list(APPEND _iq4_spv ${_out})
  endforeach()
endforeach()
add_custom_target(spv_iq4nl ALL DEPENDS ${_iq4_spv})

add_executable(vk_iq4nl iq4nl_probe.cpp)
target_link_libraries(vk_iq4nl PRIVATE Vulkan::Vulkan)
target_compile_definitions(vk_iq4nl PRIVATE
  STRATA_SPV_DIR="${CMAKE_BINARY_DIR}"
  STRATA_IQ4NL_BIN="${CMAKE_CURRENT_SOURCE_DIR}/iq4nl.bin")
if(STRATA_GLSLC_HAS_DOT)
  target_compile_definitions(vk_iq4nl PRIVATE STRATA_GLSLC_HAS_DOT=1)
endif()
add_dependencies(vk_iq4nl spv_iq4nl)
