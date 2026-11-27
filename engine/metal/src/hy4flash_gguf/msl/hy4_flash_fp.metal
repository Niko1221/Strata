// engine/metal/src/hy4flash_gguf/msl/hy4_flash_fp.metal - the floating-point mode of the WHOLE hy4-flash library.  First in
// hy4_flash_kernel_files(), so it covers the shared kernels too.
//
// `#pragma METAL fp contract(off)` is lexically scoped: it binds the expressions written after it in the
// translation unit, and a template's body counts as written where it is DEFINED, not where it is instantiated.
// hy4_flash_common.metal therefore could not reach units.metal's `unit_combine`, whose return expression
//
//     sc.x * (s0 + s1) + sc.y * (s2 + s3) - (sc.z * xsum[0] + sc.w * xsum[1])
//
// is the last step of every GEMV-shaped kernel in this engine - the dense projections, the LM head and the routed
// experts alike.  Left contractable, that step is whatever the compiler chooses per NTOK specialisation, which
// makes T invariance a measurement rather than a property.  This file is placed before quant.metal so the pragma
// reaches every line of the library.
//
// The shared kernels' own per-unit products are written as explicit fma() and are unaffected: contract(off) stops
// the COMPILER from fusing a * b + c, it does not undo an fma the source asked for.
//
// HY4_FLASH_FP_CONTRACT exists for one caller only: hy4_contract_test compiles this same library with "fast" and
// compares a dense GEMV bitwise, which is how the claim "the pragma reaches units.metal" is checked instead of
// asserted.  Production never defines it.
#ifndef HY4_FLASH_FP_CONTRACT
#define HY4_FLASH_FP_CONTRACT "METAL fp contract(off)"
#endif
_Pragma(HY4_FLASH_FP_CONTRACT)
