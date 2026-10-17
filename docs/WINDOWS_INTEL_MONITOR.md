# Intel GPU monitoring on Windows via IGCL

The SYCL server can read Intel GPU telemetry from an optional native helper.
Build and install the helper explicitly; inference does not require it.
The helper loads the driver's ControlLib.dll from the Windows system directory.

Obtain `igcl_api.h` from Intel's
[GPU Control Library](https://github.com/intel/drivers.gpu.control-library).
The tested header revision is `b6c462933502e13d1537dd5024949a51be30e63d`.
From an x64 Visual Studio developer PowerShell:

```powershell
./sycl/tools/build-windows-telemetry.ps1 -IgclIncludeDir C:/SDK/igcl/include
./build-sycl-windows/strata-intel-telemetry.exe --samples 3
```

The SYCL server discovers the helper in `build-sycl-windows`, or at the path
specified by `STRATA_INTEL_TELEMETRY_EXE`. `STRATA_INTEL_PCI_BDF` selects an
adapter by decimal bus:device.function, for example `3:0.0`.

Readings include GPU activity, device memory allocation, GPU and memory
temperature, GPU chip power, and PCIe generation/width where supported.
Chip power is derived from energy deltas and is not total board power.
PCIe link information is not traffic throughput. Unsupported fields stay empty.
Missing, exited or stale helpers do not block inference. Restart the server
after resolving a failed helper; automatic helper restart is not implemented.

Validated on Windows 10 with one Intel Arc Pro B70: real API readings under
inference load, collector unit tests, monitor tests and JavaScript label tests.
Observed peak readings included 100% activity, 119 W chip power and 64 C GPU
temperature. Helper cleanup after parent exit was also checked.
Windows 11 and multi-GPU mapping remain untested. Linux behavior has not been
revalidated by execution. The shared CUDA/HIP server is not modified.
