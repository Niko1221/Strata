# Inkling-2 Mini engine — attribution

The Inkling-2 Mini inference engine in `engine/src/inkling2/` and
`engine/include/strata/inkling2/` is based on `strata-inkling2` from
**kuraneko1/Strata** (https://github.com/kuraneko1/Strata), by
**kuraneko1 (_ryu15_)**.

That implementation is a CUDA + AVX2 hybrid engine for the `inkling2`
architecture (Thinking Machines Inkling-2 Mini), with long-context support
(65K-262K on a 12 GB RTX 4070, up to 1M context on a 32 GB V100), expert
VRAM/PCIe offloading, decode CUDA Graphs and Q8 KV cache.

License: MIT (see [LICENSE](LICENSE), Copyright (c) 2026 Niko1221 and the
Strata contributors). The upstream fork project is itself an MIT-licensed
fork of Niko1221/Strata. Ported into this Hakua fork with credit to the
author; no license terms were changed.
