# Configure-only check (no compiler, no GPU): the architecture list the FP4 sm_120 file is built for. Run as
#   cmake -P engine/tests/cmake/ink_sm120a_test.cmake
# A CMAKE_CUDA_ARCHITECTURES list such as "61-real;120-real;89-real" must give 120a-real for that file (ptxas fails on plain sm_120).
get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
include("${_here}/../../src/inkling2/cuda/ink_sm120a.cmake")

set(_failed 0)
function(expect_archs expected)
  ink_sm120a_archs(_got ${ARGN})
  string(REPLACE ";" "," _in "${ARGN}")
  if(NOT "${_got}" STREQUAL "${expected}")
    message(SEND_ERROR "FAIL [${_in}] -> [${_got}], expected [${expected}]")
  else()
    message(STATUS "ok   [${_in}] -> [${_got}]")
  endif()
endfunction()

# the bare numbers (what the release scripts pass)
expect_archs("120a" 120)
expect_archs("120a" 75 80 86 89 120)
expect_archs("120a" 120a)
expect_archs("121a" 121)
# what an inherited CMAKE_CUDA_ARCHITECTURES looks like
expect_archs("120a-real" 61-real 120-real 89-real)
expect_archs("120a-real;120a-virtual" 75-real 86-real 89-real 120-real 120-virtual)
expect_archs("120a-virtual" 120-virtual)
expect_archs("120a-real" 120a-real 120-real)                 # no duplicates
expect_archs("120a;121a-real" 120 121-real 80)
# no Blackwell entry: the list is returned as it is (suffixes untouched)
expect_archs("61-real;89-virtual" 61-real 89-virtual)
expect_archs("native" native)
expect_archs("75;86" 75 86)
# entries that only look like Blackwell are not
expect_archs("1200;12" 1200 12)
