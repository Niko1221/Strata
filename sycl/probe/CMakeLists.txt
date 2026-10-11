# sycl/probe - the four decision probes: small SYCL programs the user runs on the card before the engine's design is
# trusted (README.md here).  The parent adds this directory behind STRATA_SYCL_PROBES, so the probes are built with
# the engine's compiler and flags (-fsycl, the sub-group size, the precise FP model, the AOT target).  Nothing is
# linked beyond what the parent's options give: the SYCL runtime comes from -fsycl on the link line, the engine's
# headers from strata_includes (doorbell_probe's variant 3 is sycl_doorbell.hpp's own sys_load).  `probes` builds
# all four.
set(_probes doorbell_probe alloc_probe bw_probe graph_probe)
foreach(p IN LISTS _probes)
  add_executable(${p} ${p}.cpp probe_common.hpp)
  target_link_libraries(${p} PRIVATE strata_includes)
  if(WIN32)
    # <windows.h> without min/max macros and without the shell; getenv / fopen without MSVC's C4996 advice
    target_compile_definitions(${p} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    # GetProcessMemoryInfo (probe_common.hpp): the host-memory leak measurement of graph_probe
    target_link_libraries(${p} PRIVATE psapi)
  endif()
endforeach()
add_custom_target(probes)
add_dependencies(probes ${_probes})
