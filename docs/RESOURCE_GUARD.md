# Optional RAM and VRAM guard

When other applications consume memory, a configuration that fitted at startup can
start paging or fail later. This optional supervisor checks available physical RAM
and the selected NVIDIA GPU's actual free VRAM before launch and periodically while
the server loads and runs. It refuses startup or stops only the server it launched
if a configured floor is crossed, or a required measurement becomes unavailable.

From the repository root, using the Python environment installed for Strata:

```text
python -m serve.resource_guard --ram-floor-mib 2048 --vram-floor-mib 250 --gpu 0 -- --engine strata --config strata.json --port 8080 --open
```

These are example floors, not recommendations for every computer. Both defaults
are zero (disabled); at least one must be enabled. `--interval` defaults to one
second. `--gpu` uses the physical NVIDIA device index and overrides the server's
GPU selection, keeping monitoring and inference on the same card. RAM means
available physical memory, not free virtual address space or committed memory.

The wrapper inherits the console and accepts the normal server options after `--`.
It does not change model settings, evict other applications, or search for existing
servers. The normal server port check still applies. On Windows the worker joins
an owned Job Object before it spawns the frontend, so the frontend and engine are
contained even during startup. On POSIX it owns a new process group. Cleanup also
removes descendants left behind after the frontend exits.

Exit status: the frontend's status on normal exit, 2 for refused admission, 3 for
a runtime memory/telemetry violation, 130 for a keyboard interrupt. Startup errors
are reported as errors. A guard trip forcibly terminates its process tree: save
agent progress outside it and expect an in-flight request to fail. There is no
automatic restart.

This is a sampled pressure guard, not an allocation reservation, context saver or
guarantee against OOM. A rapid allocation can exhaust memory between samples.
Windows Job Objects and POSIX groups do not claim control over privileged children
that deliberately escape containment. The runtime has no additional pip dependency;
it reuses Strata's telemetry helpers, with psutil when available and OS RAM fallback.

Run the synthetic checks without model weights:

```text
python -m unittest serve.test_resource_guard -v
```
