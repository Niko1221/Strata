# Which architectures the HIP build compiles for (included by hip_backend.cmake; cmake/tests/hip_arch_select_test.cmake
# drives it through real configures, no HIP toolchain needed).
#
# The order: CMAKE_HIP_ARCHITECTURES, else ROCm's GPU_TARGETS, else its older AMDGPU_TARGETS (ggml-hip forwards them
# the same way), else gfx1100.  Strata links hip::host, not hip::device, so GPU_TARGETS alone used to change nothing:
# -DGPU_TARGETS=gfx1201 built a gfx1100 engine without a word, which then refused the card at startup.
#
# Two calls: strata_hip_arch_select() before find_package(hip), strata_hip_arch_record() right after it.
#
# CMAKE_HIP_ARCHITECTURES counts as given when it has a value as this file starts (the cache: -D, -C, cmake-gui; or a
# toolchain file).  Strata never writes it to the cache: what it chooses is a normal variable, set before
# enable_language(HIP), so CMake's own default (rocm_agent_enumerator, cached only when the variable is not defined)
# does not run either.  So the same -D flags choose the same architectures in a new build folder and in a used one.
# Strata 0.1.43 and older cached their default, gfx1100, under the help string _STRATA_HIP_OLD_HELP below.  That entry
# is dropped only while it still holds gfx1100; a value edited there (cmake-gui, ccmake) is kept as the user's, and
# -UCMAKE_HIP_ARCHITECTURES resets the folder to the choice below.  (A -D replaces that help string, so a -D is always kept.)
#
# GPU_TARGETS / AMDGPU_TARGETS count only when the user gave them: with GPU_TARGETS unset, find_package(hip) runs
# amdgpu-arch and caches the build machine's cards (ROCm 6.2 and 7.x), and ROCm 5.7 and 6.2 declare both variables
# with set(... CACHE STRING ...), which gives an untyped -DGPU_TARGETS=... ROCm's own type and help string but keeps
# its value.  Help strings are the package's and change between releases, so they are not read.  Instead, in order:
#   0. A normal variable (a toolchain file, a parent project) hides the cache entry, as it does everywhere in CMake: it
#      counts, also when its value equals the entry's.  ${var} reads a normal variable before the cache, so the entry
#      gets a probe value for one read and its value back right after (set_property(CACHE ... VALUE) keeps its type,
#      help string and ADVANCED; checked on CMake 3.24 to 4.4).  With no cache entry, a normal variable counts too.
#   1. CMake marks every -D: an untyped -D leaves the entry UNINITIALIZED, and any -D (typed or not, new value or the
#      same one, CMake 3.24 to 4.4 checked) sets the help string "No help, variable specified on the command line.".
#      A package can retype the entry later, but not before strata_hip_arch_select() reads it.
#   2. strata_hip_arch_record() keeps the cached value as find_package(hip) left it (STRATA_HIP_<var>_SEEN) and the
#      decision for it (STRATA_HIP_<var>_USER).  A later configure that finds the same value and no -D mark reuses
#      that decision: ROCm 5.7's retyping keeps the user's GPU_TARGETS, and amdgpu-arch's list stays ignored.  A
#      different value was edited by the user (cmake-gui, ccmake, -C) and counts; -U<var> clears it.
#      The decision recorded is the one for the cache entry alone, never a rule-0 value: taking a toolchain line out
#      returns the folder to its earlier choice, also when find_package(hip) cached the same value.
#   3. Without those records, an entry with no -D mark is ignored only in a build folder a Strata without this file
#      configured for HIP (0.1.43 or older): a cache was loaded (CMAKE_CACHEFILE_DIR, never set on a first configure
#      or with --fresh) and find_package(hip) found hip there (hip_DIR), yet strata_hip_arch_record(), which runs right
#      after it, left no records.  Anywhere else (a new folder with -C or a preset, also one that seeds hip_DIR) it
#      counts.
#
# Output (normal variables in the caller's scope): CMAKE_HIP_ARCHITECTURES when Strata chooses it, and
# STRATA_HIP_ARCH_FROM, the source the configure line names.

set(_STRATA_HIP_CLI_HELP "No help, variable specified on the command line.")
set(_STRATA_HIP_OLD_HELP "Strata HIP target architecture(s), e.g. gfx1100 or gfx1100;gfx1201")

# "gfx1100 gfx1201" or "gfx1100;gfx1201;gfx1100" -> gfx1100;gfx1201
function(_strata_hip_arch_norm in out)
  string(REPLACE " " ";" _v "${in}")
  list(REMOVE_ITEM _v "")
  list(REMOVE_DUPLICATES _v)
  set(${out} "${_v}" PARENT_SCOPE)
endfunction()

# The targets the user gave in <var> (GPU_TARGETS or AMDGPU_TARGETS), empty when none.  <legacy>: rule 3's folder.
# Leaves the decision for the cache entry (rules 1-3, without rule 0) for strata_hip_arch_record().
function(_strata_hip_user_targets var legacy out)
  set(_seen STRATA_HIP_${var}_SEEN)
  if(NOT DEFINED CACHE{${var}})
    set(_v "${${var}}")                         # 0. a toolchain file's or a parent project's variable, or nothing
    set(_cached "")
  else()
    # 0. Is there a normal variable?  It hides the entry whatever its value, so give the entry a value no user writes,
    # read ${var}, and put the value back before anything else reads the cache.
    get_property(_value CACHE ${var} PROPERTY VALUE)
    set(_probe "strata-hip-arch-select-probe-${var}")
    set_property(CACHE ${var} PROPERTY VALUE "${_probe}")
    set(_visible "${${var}}")
    set_property(CACHE ${var} PROPERTY VALUE "${_value}")
    get_property(_type CACHE ${var} PROPERTY TYPE)
    get_property(_help CACHE ${var} PROPERTY HELPSTRING)
    if("${_type}" STREQUAL "UNINITIALIZED" OR "${_help}" STREQUAL "${_STRATA_HIP_CLI_HELP}")
      set(_cached "${_value}")                  # 1. a -D, on this configure or an earlier one
    elseif(DEFINED CACHE{${_seen}})
      if("${_value}" STREQUAL "$CACHE{${_seen}}")
        set(_cached "$CACHE{STRATA_HIP_${var}_USER}")   # 2. unchanged since the last configure: the same decision
      else()
        set(_cached "${_value}")                # 2. edited since the last configure
      endif()
    elseif(legacy)
      set(_cached "")                           # 3. a 0.1.43 folder: maybe the package's
    else()
      set(_cached "${_value}")                  # 3. no Strata HIP configure here before: -C, a preset
    endif()
    if(NOT "${_visible}" STREQUAL "${_probe}")
      set(_v "${_visible}")                     # 0. a normal variable hides the cache entry
    else()
      set(_v "${_cached}")
    endif()
  endif()
  _strata_hip_arch_norm("${_v}" _v)
  _strata_hip_arch_norm("${_cached}" _cached)
  set_property(GLOBAL PROPERTY STRATA_HIP_${var}_DECIDED "${_cached}")
  set(${out} "${_v}" PARENT_SCOPE)
endfunction()

function(strata_hip_arch_select)
  # rule 3: a folder configured for HIP by a Strata without this file
  set(_legacy FALSE)
  if(DEFINED CACHE{CMAKE_CACHEFILE_DIR} AND DEFINED CACHE{hip_DIR}
     AND NOT DEFINED CACHE{STRATA_HIP_GPU_TARGETS_SEEN} AND NOT DEFINED CACHE{STRATA_HIP_AMDGPU_TARGETS_SEEN})
    if(NOT "$CACHE{hip_DIR}" MATCHES "-NOTFOUND$")
      set(_legacy TRUE)
    endif()
  endif()
  if(DEFINED CACHE{CMAKE_HIP_ARCHITECTURES})
    get_property(_help CACHE CMAKE_HIP_ARCHITECTURES PROPERTY HELPSTRING)
    _strata_hip_arch_norm("$CACHE{CMAKE_HIP_ARCHITECTURES}" _old)
    if("${_help}" STREQUAL "${_STRATA_HIP_OLD_HELP}" AND "${_old}" STREQUAL "gfx1100")
      unset(CMAKE_HIP_ARCHITECTURES CACHE)      # an older Strata's own default, not the user's
    endif()
  endif()
  _strata_hip_user_targets(GPU_TARGETS ${_legacy} _gpu)
  _strata_hip_user_targets(AMDGPU_TARGETS ${_legacy} _amdgpu)
  if(NOT "${CMAKE_HIP_ARCHITECTURES}" STREQUAL "")
    set(_from "CMAKE_HIP_ARCHITECTURES")
    set(_theirs "")
    if(_gpu)
      set(_name GPU_TARGETS)
      set(_theirs ${_gpu})
    elseif(_amdgpu)
      set(_name AMDGPU_TARGETS)
      set(_theirs ${_amdgpu})
    endif()
    if(_theirs)
      _strata_hip_arch_norm("${CMAKE_HIP_ARCHITECTURES}" _mine)
      list(SORT _mine)
      list(SORT _theirs)
      if(NOT _mine STREQUAL _theirs)
        message(WARNING "Strata HIP: CMAKE_HIP_ARCHITECTURES (${CMAKE_HIP_ARCHITECTURES}) and ${_name} "
          "(${_theirs}) differ; building ${CMAKE_HIP_ARCHITECTURES}, the CMAKE_HIP_ARCHITECTURES value "
          "(-UCMAKE_HIP_ARCHITECTURES drops it from this build folder)")
      endif()
    endif()
  else()
    if(_gpu)
      set(_arch "${_gpu}")
      set(_from "GPU_TARGETS")
    elseif(_amdgpu)
      set(_arch "${_amdgpu}")
      set(_from "AMDGPU_TARGETS")
    else()
      set(_arch gfx1100)
      set(_from "the default; pass -DCMAKE_HIP_ARCHITECTURES or -DGPU_TARGETS for another card")
    endif()
    set(CMAKE_HIP_ARCHITECTURES "${_arch}" PARENT_SCOPE)
  endif()
  set(STRATA_HIP_ARCH_FROM "${_from}" PARENT_SCOPE)
endfunction()

# After find_package(hip): the cached values as the package left them, and the decision for each (rule 2).
# $CACHE{}, not the variables: ROCm 7.x sets a normal GPU_TARGETS from AMDGPU_TARGETS, which does not outlive this
# configure.
function(strata_hip_arch_record)
  foreach(_var IN ITEMS GPU_TARGETS AMDGPU_TARGETS)
    get_property(_decided GLOBAL PROPERTY STRATA_HIP_${_var}_DECIDED)
    set(STRATA_HIP_${_var}_SEEN "$CACHE{${_var}}" CACHE INTERNAL
        "${_var} in the cache after find_package(hip) (cmake/hip_arch_select.cmake)")
    set(STRATA_HIP_${_var}_USER "${_decided}" CACHE INTERNAL
        "the decision for that ${_var} entry (cmake/hip_arch_select.cmake)")
  endforeach()
endfunction()
