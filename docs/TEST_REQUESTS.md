# Test requests and hardware wanted

Strata is built and measured on a few machines. Results from other hardware are welcome and are used: they tell us
what a setup really does and show where the engine picks something wrong. Write a report as described in
[COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md): the build, the card and PCIe width, the CPU and RAM speed, the
model file, the exact command, three runs per size, and what you did not test. Post it in an issue, or on the pull
request it is about.

This list is edited by the maintainers. An item stays until a report answers it.

| Wanted | Why | Example of an earlier answer |
|---|---|---|
| Two NVIDIA cards of different sizes (a large and a small one), in both card orders, with `--layer-split auto` | There is no unequal pair here to check the split planner; the card order and the small card's free memory changed the split | #880, #1238 |
| Three or four NVIDIA GPUs | No multi-GPU NVIDIA rig of that size here | #880 |
| Pascal (sm_61: Tesla P40, P100, GTX 10 series) | Older cards run on the fp32 path and can behave differently from newer ones | #875, #876 |
| A Linux host with 32 GB of RAM or less | The expert file tier and the resident mode depend on how much of the model fits in RAM | #1194 |
| A long soak (an hour or more of back-to-back requests) on a split or resident setup | Shows crashes, drift and heat that a short run does not | #848 |

To add a request, open a pull request that adds a row, or tell the maintainers in an issue.
