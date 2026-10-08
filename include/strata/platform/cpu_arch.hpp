#pragma once

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define STRATA_CPU_X86 1
#else
#define STRATA_CPU_X86 0
#endif
