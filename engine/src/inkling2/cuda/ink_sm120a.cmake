# The FP4 block-scaled MMA of ink_moe_fp4_sm120.cu needs the arch-specific targets sm_120a / sm_121a (plain sm_120 makes ptxas fail:
# "mma with block scale not supported on sm_120"). This maps a CUDA architecture list to the list that translation unit is built for.
#
# An entry is a number with an optional "a" and an optional "-real" / "-virtual" suffix, as CMAKE_CUDA_ARCHITECTURES writes it
# ("61-real;120-real;89-real" is what a build that passes CMAKE_CUDA_ARCHITECTURES without STRATA_INK_CUDA_ARCHS inherits). The suffix is kept:
# 120 -> 120a, 120-real -> 120a-real, 120-virtual -> 120a-virtual, 121 -> 121a. Entries for other cards drop out; a list without a Blackwell
# entry is returned as it is (the file then builds, to stubs, for the cards it is given).
function(ink_sm120a_archs out_var)
  set(_result "")
  foreach(_entry IN LISTS ARGN)
    if("${_entry}" MATCHES "^(12[01])a?(-real|-virtual)?$")
      list(APPEND _result "${CMAKE_MATCH_1}a${CMAKE_MATCH_2}")
    endif()
  endforeach()
  if("${_result}" STREQUAL "")
    set(_result ${ARGN})
  endif()
  list(REMOVE_DUPLICATES _result)
  set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()
