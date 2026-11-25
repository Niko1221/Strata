# STRATA_KEEP_SYMBOLS: compile the HOST code with debug information so the release scripts can keep a symbol file
# (PDB, .debug, dSYM) for every engine they ship.  The symbol files are never part of a package or a release asset
# (tools/packaging/keep_symbols.py, docs/RELEASING.md); they are what turns a crash address or a minidump from a
# user's engine back into a source line.
#
# The generated code does not change: debug information is a separate section, and no flag here lowers an
# optimisation level or defines a macro.  Device code never gets debug flags (nvcc -G and clang's device -g do
# turn optimisations off); only the host side of a .cu / .hip file does.  tools/packaging/keep_symbols.py check
# compares the code sections (.text, .nv_fatbin on ELF; .text on PE) of a build made with and without it.
#   MSVC:   /Z7 (the information lives in the .obj files, so parallel cl.exe and nvcc runs never share a .pdb), and the
#           linker's /DEBUG with /OPT:REF /OPT:ICF given again, because /DEBUG alone turns the default folding off
#           (that WOULD change the code).  /PDBALTPATH keeps the build folder out of the exe.
#   others: -g (clang and gcc: host code only, `-Xarch_host` for the HIP and SYCL device passes).
include_guard(GLOBAL)
option(STRATA_KEEP_SYMBOLS "compile host code with debug information so a symbol file can be kept per release build (same machine code)" OFF)
if(STRATA_KEEP_SYMBOLS)
  if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/Z7>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Z7>")
    add_link_options("LINKER:/DEBUG" "LINKER:/OPT:REF" "LINKER:/OPT:ICF" "LINKER:/INCREMENTAL:NO" "LINKER:/PDBALTPATH:%_PDB%")
  elseif(WIN32)
    # clang for Windows (the ROCm HIP build, MSVC target): CodeView through -g, linked by lld-link
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-g>" "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Xarch_host -g>")
    add_link_options("LINKER:/DEBUG" "LINKER:/OPT:REF" "LINKER:/OPT:ICF" "LINKER:/INCREMENTAL:NO" "LINKER:/PDBALTPATH:%_PDB%")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    # icpx -fsycl: -g would reach the device compile too (and the AOT backend); the host pass only
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:SHELL:-Xarch_host -g>")
  else()
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-g>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -g>"
                        "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Xarch_host -g>")
  endif()
  message(STATUS "STRATA_KEEP_SYMBOLS: host debug information on (the code does not change)")
endif()
