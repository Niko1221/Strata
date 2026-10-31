# P100 VMAD accumulator alias regression

At upstream 4b9e05a2a806a3fe24f7b2560d98626c37dade35, the shared
`STRATA_DP4A` wrapper already uses VMAD on SM60 and native DP4A on SM61+.
This patch preserves that architecture selection. It fixes a separate
correctness error in SM60's four-instruction inline assembly.

The deployed CUDA 12 engine 0.1.48 on paulos was also inspected read-only.
Its binary SHA256 is
`8fb3291b8ee809a4dc357ad538ba92a02eef2bc0e520b6dbd1cadc0c3df5c544`.
`cuobjdump --gpu-architecture sm_60 --dump-sass` counted 385896 VMAD
instructions and zero IDP instructions. This confirms that binary does not
force native DP4A on P100. The alias regression below was reproduced against
the current upstream header, not as a production response-corruption test.

The `+r` accumulator constraint allowed a compiler to share its register with
an equal-valued input. The first instruction overwrote that input before the
next three instructions read its remaining bytes. `+&r` marks the accumulator
as early-clobbered and prevents that sharing.

The regression probe explicitly calls the wrapper with accumulator equal to
input A, input B, and both inputs. Its CPU oracle accumulates modulo 2^32,
avoiding signed-overflow undefined behavior.

```sh
nvcc -O3 -arch=sm_60 -Iinclude tools/sm60_dp4a_alias_check.cu -o alias-check
./alias-check
cuobjdump --gpu-architecture sm_60 --dump-sass alias-check
```

CUDA 12.8.1 on paulos: before the fix, one P100 returned
62997, 62238, 61325 and 61324 mismatches in the four arms (65536 inputs per
arm). After the fix, all four arms returned zero mismatches on each of three
P100s, an RTX 2080 Ti and an RTX 3060: 262144 comparisons per card. Probes
compiled for SM60, SM61, SM75 and SM86. The SM60 probe's SASS contains no IDP
instructions; the SM61 probe retains four native IDP instructions.

This is a focused arithmetic regression, not a whole-engine generation or
throughput benchmark. No deployment or performance default was changed.
The CUDA-only SM60 branch is the only modified code path; HIP/SYCL builds and
a complete engine build have not been run for this patch.
