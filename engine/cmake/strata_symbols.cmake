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
#   others: -g on gcc and clang C/C++ files and on nvcc's host compiler; icpx: -Xarch_host -g1; AppleClang: -gline-tables-only; HIP-language files: nothing (a flag there changed the
#           code).  Each is shown with a same-build-directory control (tools/packaging/keep_symbols.py check) by the release scripts' owners.
include_guard(GLOBAL)
option(STRATA_KEEP_SYMBOLS "compile host code with debug information so a symbol file can be kept per release build (same machine code)" OFF)
if(STRATA_KEEP_SYMBOLS)
  if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/Z7>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Z7>")
    add_link_options("LINKER:/DEBUG" "LINKER:/OPT:REF" "LINKER:/OPT:ICF" "LINKER:/INCREMENTAL:NO" "LINKER:/PDBALTPATH:%_PDB%")
  elseif(WIN32)
    # clang for Windows (the ROCm HIP build, MSVC target): CodeView line tables, linked by lld-link
    # (HIP-language files get no flag: -Xarch_host -g changed .text, .hip_fatbin and .rodata of the Linux HIP build, measured with a same-directory control;
    # and plain -g changed the instruction selection of expert_source.cpp, 10 of 108,318 lines, in the Windows build: -gline-tables-only and -g1 give
    # assembly identical to no flag on that file and on three more, so the C and C++ files take -gline-tables-only)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-gline-tables-only>")
    add_link_options("LINKER:/DEBUG" "LINKER:/OPT:REF" "LINKER:/OPT:ICF" "LINKER:/INCREMENTAL:NO" "LINKER:/PDBALTPATH:%_PDB%")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    # icpx -fsycl: -g would reach the device compile too (and the AOT backend), and -Xarch_host -g changed the code: the host pass with line tables
    # only (-g1) is byte-equal
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:SHELL:-Xarch_host -g1>")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    # clang -g changes instruction selection a little: line tables only (the Mac release script keeps it only if __text is byte-equal, tools/packaging/mac/build_engine.sh)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-gline-tables-only>")
  else()
    # gcc and clang host code, nvcc's host compiler.  No HIP-language flag (see above): a HIP file's host side gets no debug information, the
    # C++ files of the engine do.
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-g>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -g>")
  endif()
  message(STATUS "STRATA_KEEP_SYMBOLS: host debug information on (the code does not change)")
endif()
