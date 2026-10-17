# Which architectures a HIP build compiles for (cmake/hip_arch_select.cmake), over new and reused build folders and
# the cache entries ROCm 5.7, 6.2 and 7.2 write.  Each step is a real configure of the small project in
# hip_arch_select/, which replays ROCm's find_package(hip) declarations: no HIP compiler, no ROCm, no GPU.
#
#   cmake [-DGENERATOR=Ninja] [-DMAKE_PROGRAM=...] [-DWORK_DIR=...] [-DKEEP=ON] -P cmake/tests/hip_arch_select_test.cmake
#
# (every -D before -P: cmake ignores the ones after it)
#
# ctest runs it as hip_arch_select_test (STRATA_BUILD_TESTS).  Each run makes its own folder,
# strata_hip_arch_select_test-<time>-<random>, inside WORK_DIR (default: the system's temporary folder), so runs at the
# same time do not share build folders.  It is deleted when every step passes (unless KEEP) and kept for a look at
# the caches when one fails.
cmake_minimum_required(VERSION 3.24)

get_filename_component(_src "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(_proj "${CMAKE_CURRENT_LIST_DIR}/hip_arch_select")
if(NOT WORK_DIR)
  foreach(_tmp IN ITEMS "$ENV{TMPDIR}" "$ENV{TEMP}" "$ENV{TMP}" "/tmp")
    if(NOT _tmp STREQUAL "" AND IS_DIRECTORY "${_tmp}")
      set(WORK_DIR "${_tmp}")
      break()
    endif()
  endforeach()
endif()
string(TIMESTAMP _stamp "%Y%m%d-%H%M%S" UTC)
string(RANDOM LENGTH 8 ALPHABET "0123456789abcdef" _rand)
set(WORK_DIR "${WORK_DIR}/strata_hip_arch_select_test-${_stamp}-${_rand}")
if(EXISTS "${WORK_DIR}")
  message(FATAL_ERROR "hip_arch_select_test: ${WORK_DIR} exists already")
endif()
file(MAKE_DIRECTORY "${WORK_DIR}")
set(_gen "")
if(GENERATOR)
  list(APPEND _gen -G "${GENERATOR}")
endif()
if(MAKE_PROGRAM)
  list(APPEND _gen "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
set(_cards "gfx1201\;gfx1201")   # what amdgpu-arch prints on a machine with two gfx1201 cards
set(_failures 0)
set(_steps 0)

# step(<folder> <rocm> <expected arch> <expected source> <WARN|NOWARN> [cmake args...])
# Configures WORK_DIR/<folder> (kept between steps, as a user's build folder is) and checks what it chose.
# A ';' inside an argument is written '\;'.
function(step dir rocm want_arch want_from want_warn)
  math(EXPR _n "${_steps} + 1")
  set(_steps ${_n} PARENT_SCOPE)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${_proj}" -B "${WORK_DIR}/${dir}" ${_gen}
            "-DSTRATA_SOURCE_DIR=${_src}" "-DSIM_ROCM=${rocm}" -DSIM_CARDS=${_cards} ${ARGN}
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  string(REPLACE "\\;" ";" _args "${ARGN}")
  string(REPLACE ";-" " -" _args "${_args}")
  set(_what "${dir} (ROCm ${rocm}) ${_args}")
  if(NOT _rc EQUAL 0)
    message(SEND_ERROR "FAIL ${_what}: configure failed\n${_out}\n${_err}")
    math(EXPR _f "${_failures} + 1")
    set(_failures ${_f} PARENT_SCOPE)
    return()
  endif()
  file(READ "${WORK_DIR}/${dir}/arch.txt" _raw)
  string(FIND "${_raw}" "|" _bar)
  string(SUBSTRING "${_raw}" 0 ${_bar} _got)
  math(EXPR _bar "${_bar} + 1")
  string(SUBSTRING "${_raw}" ${_bar} -1 _got_from)
  string(REGEX MATCH "^[^ ;]*" _got_from_word "${_got_from}")   # "the default; pass ..." -> "the"
  if(_err MATCHES "Strata HIP: CMAKE_HIP_ARCHITECTURES \\([^)]*\\) and[ \n]+[A-Z_]+[ \n]+\\([^)]*\\)[ \n]+differ")   # CMake wraps it
    set(_warned WARN)
  else()
    set(_warned NOWARN)
  endif()
  set(_cached "")
  file(STRINGS "${WORK_DIR}/${dir}/CMakeCache.txt" _cached REGEX "^CMAKE_HIP_ARCHITECTURES[:=]")
  string(REPLACE "\\;" ";" _want_arch "${want_arch}")
  set(_ok TRUE)
  if(NOT "${_got}" STREQUAL "${_want_arch}" OR NOT "${_got_from_word}" STREQUAL "${want_from}"
     OR NOT _warned STREQUAL want_warn)
    set(_ok FALSE)
  endif()
  # Strata chose: it must not have put its choice in the cache, where the next configure would take it for the user's
  # (the stand-in for Strata 0.1.43 did, as 0.1.43 does)
  if(NOT want_from MATCHES "^(CMAKE_HIP_ARCHITECTURES|Strata)$" AND _cached)
    set(_ok FALSE)
  endif()
  if(_ok)
    message(STATUS "ok   ${_what}: ${_got} from ${_got_from_word}, ${_warned}")
  else()
    message(SEND_ERROR "FAIL ${_what}: got ${_got} from '${_got_from}', ${_warned}, cache '${_cached}'; "
      "want ${_want_arch} from ${want_from}, ${want_warn}, no CMAKE_HIP_ARCHITECTURES in the cache unless given\n${_err}")
    math(EXPR _f "${_failures} + 1")
    set(_failures ${_f} PARENT_SCOPE)
  endif()
endfunction()

# edit_cache(<folder> <from> <to>): change a line of WORK_DIR/<folder>/CMakeCache.txt, as cmake-gui or ccmake would
function(edit_cache dir from to)
  file(READ "${WORK_DIR}/${dir}/CMakeCache.txt" _cache)
  string(REPLACE "${from}" "${to}" _edited "${_cache}")
  if(_edited STREQUAL _cache)
    message(SEND_ERROR "FAIL ${dir}: no '${from}' line in the cache to edit")
    math(EXPR _f "${_failures} + 1")
    set(_failures ${_f} PARENT_SCOPE)
  endif()
  file(WRITE "${WORK_DIR}/${dir}/CMakeCache.txt" "${_edited}")
endfunction()

foreach(rocm IN ITEMS 7.2 6.2 5.7)
  string(REPLACE "." "" r "${rocm}")
  # each source in a new build folder, then the same folder configured again with no flags
  step(gpu${r} ${rocm} gfx1201 GPU_TARGETS NOWARN -DGPU_TARGETS=gfx1201)
  step(gpu${r} ${rocm} gfx1201 GPU_TARGETS NOWARN)   # 5.7 / 6.2 have retyped GPU_TARGETS under their help string
  step(gpu${r} ${rocm} gfx1201 GPU_TARGETS NOWARN)
  step(amd${r} ${rocm} gfx1201 AMDGPU_TARGETS NOWARN -DAMDGPU_TARGETS=gfx1201)
  step(amd${r} ${rocm} gfx1201 AMDGPU_TARGETS NOWARN)   # 5.7 / 6.2 copied it into GPU_TARGETS: still not the user's
  step(hip${r} ${rocm} gfx1201 CMAKE_HIP_ARCHITECTURES NOWARN -DCMAKE_HIP_ARCHITECTURES=gfx1201)
  step(hip${r} ${rocm} gfx1201 CMAKE_HIP_ARCHITECTURES NOWARN)   # amdgpu-arch cached gfx1201;gfx1201: a set, no warning
  step(none${r} ${rocm} gfx1100 the NOWARN)
  step(none${r} ${rocm} gfx1100 the NOWARN)   # the machine's cards, cached by 6.2 / 7.2, are not the user's
  # both given, in a new build folder and again with no flags: CMAKE_HIP_ARCHITECTURES wins and the warning says so
  step(both${r} ${rocm} gfx1100 CMAKE_HIP_ARCHITECTURES WARN -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1201)
  step(both${r} ${rocm} gfx1100 CMAKE_HIP_ARCHITECTURES WARN)
  # the same flags in the folder Strata configured with nothing given (it chose gfx1100 there): the same result
  step(none${r} ${rocm} gfx1100 CMAKE_HIP_ARCHITECTURES WARN -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1201)
  # a used folder: -DGPU_TARGETS chooses, a configure with no flags keeps it, -U drops it
  step(reuse${r} ${rocm} gfx1100 the NOWARN)
  step(reuse${r} ${rocm} gfx1201 GPU_TARGETS NOWARN -DGPU_TARGETS=gfx1201)
  step(reuse${r} ${rocm} gfx1201 GPU_TARGETS NOWARN)
  step(reuse${r} ${rocm} gfx1101 GPU_TARGETS NOWARN -DGPU_TARGETS=gfx1101)
  step(reuse${r} ${rocm} gfx1100 the NOWARN -UGPU_TARGETS)
  step(reuse${r} ${rocm} gfx1100 the NOWARN)
  # -DGPU_TARGETS with exactly the list amdgpu-arch cached in that folder still counts as given
  step(same${r} ${rocm} gfx1100 the NOWARN)
  step(same${r} ${rocm} "gfx1201" GPU_TARGETS NOWARN "-DGPU_TARGETS=${_cards}")
  step(same${r} ${rocm} "gfx1201" GPU_TARGETS NOWARN)
  # lists compare as sets; a list that differs warns and names the variable
  step(set${r} ${rocm} "gfx1201\;gfx1100" CMAKE_HIP_ARCHITECTURES NOWARN
       "-DCMAKE_HIP_ARCHITECTURES=gfx1201\;gfx1100" "-DGPU_TARGETS=gfx1100 gfx1201")
  step(diff${r} ${rocm} gfx1201 CMAKE_HIP_ARCHITECTURES WARN
       -DCMAKE_HIP_ARCHITECTURES=gfx1201 "-DGPU_TARGETS=gfx1100\;gfx1201")
  step(damd${r} ${rocm} gfx1201 CMAKE_HIP_ARCHITECTURES WARN -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DAMDGPU_TARGETS=gfx1100)
  # GPU_TARGETS from an initial cache (-C), not a -D
  file(WRITE "${WORK_DIR}/init.cmake" "set(GPU_TARGETS gfx1201 CACHE STRING \"the R9700\")\n")
  step(init${r} ${rocm} gfx1201 GPU_TARGETS NOWARN -C "${WORK_DIR}/init.cmake")
  step(init${r} ${rocm} gfx1201 GPU_TARGETS NOWARN)
  # a build folder from Strata 0.1.43, which cached its gfx1100: no longer taken for the user's choice
  step(old${r} ${rocm} gfx1100 Strata NOWARN -DSIM_OLD_STRATA=ON)
  step(old${r} ${rocm} gfx1100 the NOWARN -DSIM_OLD_STRATA=OFF)
  step(old${r} ${rocm} gfx1201 GPU_TARGETS NOWARN -DGPU_TARGETS=gfx1201)
  step(oldx${r} ${rocm} gfx1100 Strata NOWARN -DSIM_OLD_STRATA=ON)
  step(oldx${r} ${rocm} gfx1100 CMAKE_HIP_ARCHITECTURES WARN
       -DSIM_OLD_STRATA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1201)
  # a 0.1.43 folder whose CMAKE_HIP_ARCHITECTURES was edited in cmake-gui (0.1.43's help string stays): the user's,
  # kept over reconfigures; -UCMAKE_HIP_ARCHITECTURES resets it
  step(olde${r} ${rocm} gfx1100 Strata NOWARN -DSIM_OLD_STRATA=ON)
  edit_cache(olde${r} "CMAKE_HIP_ARCHITECTURES:STRING=gfx1100" "CMAKE_HIP_ARCHITECTURES:STRING=gfx1201")
  step(olde${r} ${rocm} gfx1201 CMAKE_HIP_ARCHITECTURES NOWARN -DSIM_OLD_STRATA=OFF)
  step(olde${r} ${rocm} gfx1201 CMAKE_HIP_ARCHITECTURES NOWARN)
  step(olde${r} ${rocm} gfx1201 CMAKE_HIP_ARCHITECTURES WARN -DGPU_TARGETS=gfx1101)
  step(olde${r} ${rocm} gfx1101 GPU_TARGETS NOWARN -UCMAKE_HIP_ARCHITECTURES)
  step(oldu${r} ${rocm} gfx1100 Strata NOWARN -DSIM_OLD_STRATA=ON)
  edit_cache(oldu${r} "CMAKE_HIP_ARCHITECTURES:STRING=gfx1100" "CMAKE_HIP_ARCHITECTURES:STRING=gfx1201")
  step(oldu${r} ${rocm} gfx1100 the NOWARN -DSIM_OLD_STRATA=OFF -UCMAKE_HIP_ARCHITECTURES)
  # an initial cache that also names the ROCm package (hip_DIR) in a new folder: GPU_TARGETS still counts
  file(WRITE "${WORK_DIR}/init-hip.cmake" "set(hip_DIR \"/opt/rocm/lib/cmake/hip\" CACHE PATH \"package\")\n"
       "set(GPU_TARGETS gfx1201 CACHE STRING \"the R9700\")\n")
  step(inith${r} ${rocm} gfx1201 GPU_TARGETS NOWARN -C "${WORK_DIR}/init-hip.cmake")
  step(inith${r} ${rocm} gfx1201 GPU_TARGETS NOWARN)
  # a toolchain file's (normal) GPU_TARGETS / AMDGPU_TARGETS, added after ROCm cached its entries, wins over them;
  # taken out again, the folder is back to the default.  tc*: a value that differs from the entry.  tce*: exactly the
  # list amdgpu-arch cached, so the normal variable equals the entry (GPU_TARGETS on 7.2 and 6.2, AMDGPU_TARGETS on
  # 6.2; 7.2 caches no AMDGPU_TARGETS and 5.7 caches both empty, so there it differs or has no entry)
  foreach(var IN ITEMS GPU_TARGETS AMDGPU_TARGETS)
    foreach(kind IN ITEMS tc tce)
      if(kind STREQUAL "tc")
        set(_val gfx1101)
        set(_want gfx1101)
      else()
        string(REPLACE "\\;" ";" _val "${_cards}")
        set(_want gfx1201)
      endif()
      set(_tc "${WORK_DIR}/${kind}-${var}${r}.cmake")
      file(WRITE "${_tc}" "# nothing yet\n")
      step(${kind}${var}${r} ${rocm} gfx1100 the NOWARN "-DCMAKE_TOOLCHAIN_FILE=${_tc}")
      step(${kind}${var}${r} ${rocm} gfx1100 the NOWARN)
      file(WRITE "${_tc}" "set(${var} \"${_val}\")\n")
      step(${kind}${var}${r} ${rocm} ${_want} ${var} NOWARN)
      step(${kind}${var}${r} ${rocm} ${_want} ${var} NOWARN)
      file(WRITE "${_tc}" "# taken out\n")
      step(${kind}${var}${r} ${rocm} gfx1100 the NOWARN)
      step(${kind}${var}${r} ${rocm} gfx1100 the NOWARN)
    endforeach()
  endforeach()
  # a toolchain file's GPU_TARGETS over a -DGPU_TARGETS (CMake: the normal variable hides the cache entry); taken
  # out, the -D counts again
  set(_tc "${WORK_DIR}/tc-d${r}.cmake")
  file(WRITE "${_tc}" "set(GPU_TARGETS gfx1101)\n")
  step(tcd${r} ${rocm} gfx1101 GPU_TARGETS NOWARN "-DCMAKE_TOOLCHAIN_FILE=${_tc}" -DGPU_TARGETS=gfx1201)
  step(tcd${r} ${rocm} gfx1101 GPU_TARGETS NOWARN)
  file(WRITE "${_tc}" "# taken out\n")
  step(tcd${r} ${rocm} gfx1201 GPU_TARGETS NOWARN)
  # a toolchain file's GPU_TARGETS / AMDGPU_TARGETS from the start on a one-card machine: ROCm 6.2 caches that same
  # value (7.2 caches it for GPU_TARGETS); taken out, the folder is back to the default, not to what ROCm cached
  foreach(var IN ITEMS GPU_TARGETS AMDGPU_TARGETS)
    set(_tc "${WORK_DIR}/tc1-${var}${r}.cmake")
    file(WRITE "${_tc}" "set(${var} gfx1201)\n")
    step(tc1${var}${r} ${rocm} gfx1201 ${var} NOWARN "-DCMAKE_TOOLCHAIN_FILE=${_tc}" -DSIM_CARDS=gfx1201)
    step(tc1${var}${r} ${rocm} gfx1201 ${var} NOWARN -DSIM_CARDS=gfx1201)
    step(tc1${var}${r} ${rocm} gfx1201 ${var} NOWARN -DSIM_CARDS=gfx1201)
    file(WRITE "${_tc}" "# taken out\n")
    step(tc1${var}${r} ${rocm} gfx1100 the NOWARN -DSIM_CARDS=gfx1201)
    step(tc1${var}${r} ${rocm} gfx1100 the NOWARN -DSIM_CARDS=gfx1201)
  endforeach()
endforeach()

# GPU_TARGETS edited in the cache (cmake-gui, ccmake) since the last configure counts as the user's
step(edit 7.2 gfx1100 the NOWARN)
edit_cache(edit "GPU_TARGETS:STRING=gfx1201;gfx1201" "GPU_TARGETS:STRING=gfx1101")
step(edit 7.2 gfx1101 GPU_TARGETS NOWARN)
step(edit 7.2 gfx1101 GPU_TARGETS NOWARN)

if(_failures)
  message(FATAL_ERROR "hip_arch_select_test: ${_failures} of ${_steps} steps failed; build folders kept in ${WORK_DIR}")
endif()
if(NOT KEEP)
  file(REMOVE_RECURSE "${WORK_DIR}")
endif()
message(STATUS "hip_arch_select_test: all ${_steps} steps passed (CMake ${CMAKE_VERSION})")
