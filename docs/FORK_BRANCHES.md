# Fork branches, activation and public evidence

This fork contributes to [Niko1221/Strata](https://github.com/Niko1221/Strata).
The branches below are separate changes. Selecting a branch does not enable
features from the other branches. `main` follows upstream and does not contain
these unmerged contributions. The fork's landing page is the community hardware
branch.

## Current contributions

| Branch | Default and how to use it | Instructions and evidence |
| --- | --- | --- |
| [`feat/responses-api-451`](https://github.com/CC-David-CC/Strata-a5500/tree/feat/responses-api-451) | Responses is **off**. Start this branch's Python server with `--experimental-responses`, or set top-level `"experimental_responses": true` in its run config and restart. | [Enable, verify and disable](https://github.com/CC-David-CC/Strata-a5500/blob/feat/responses-api-451/docs/RESPONSES.md#enable-and-verify); [CUDA smoke-test evidence](https://github.com/CC-David-CC/Strata-a5500/blob/feat/responses-api-451/docs/benchmarks/2026-10-02-responses-cuda.json) |
| [`contrib/gfx1012-community`](https://github.com/CC-David-CC/Strata-a5500/tree/contrib/gfx1012-community) | Manual Linux HIP build with `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1012`; run the resulting executable. No additional runtime feature switch. The automatic installer does not select this community target. | [Build and run](COMMUNITY_GFX1012_REVIEW.md#build-and-run-focused-checks); [measurements](benchmarks/2026-10-01-community-gfx1012.md); [0.1.34 checks](benchmarks/2026-10-02-upstream-sync-hardware.md) |
| [`contrib/non-mtp-serving`](https://github.com/CC-David-CC/Strata-a5500/tree/contrib/non-mtp-serving) | Existing MTP configs keep their behavior. To serve without a drafter, remove `--mtp` and its directory from engine arguments; use `--spec 2 --conversation-cache-mib 0`. Restart with this branch's executable. | [Start without MTP](https://github.com/CC-David-CC/Strata-a5500/blob/contrib/non-mtp-serving/docs/NON_MTP_SERVING_REVIEW.md#start-the-server-without-mtp); [measurements](https://github.com/CC-David-CC/Strata-a5500/blob/contrib/non-mtp-serving/docs/benchmarks/2026-10-01-serving.md); [0.1.34 checks](https://github.com/CC-David-CC/Strata-a5500/blob/contrib/non-mtp-serving/docs/benchmarks/2026-10-02-upstream-sync-serving.md) |
| [`perf/fleet-mmvq`](https://github.com/CC-David-CC/Strata-a5500/tree/perf/fleet-mmvq) | Both alternatives are **off**. Build this branch, then set `STRATA_GR_DOWN_MAX4=1` or `STRATA_MMVQ_WARP1=1` in the engine environment. Test one at a time and restart after changing settings. | [Enable, verify and disable](https://github.com/CC-David-CC/Strata-a5500/blob/perf/fleet-mmvq/docs/EXPERIMENTAL_KERNELS.md#enable-an-option-for-serving); [measurements](https://github.com/CC-David-CC/Strata-a5500/blob/perf/fleet-mmvq/docs/benchmarks/2026-10-01-experimental-kernels.md); [0.1.34 checks](https://github.com/CC-David-CC/Strata-a5500/blob/perf/fleet-mmvq/docs/benchmarks/2026-10-02-upstream-sync-experimental.md) |

Use a checkout of the named branch. For example:

```sh
git clone --branch feat/responses-api-451 https://github.com/CC-David-CC/Strata-a5500.git Strata-responses
cd Strata-responses
```

For native changes, the run config's `exe` must point to the executable built
from that branch. An older or upstream executable cannot acquire new kernels
from environment variables alone. Responses changes the Python adapter and
uses the existing native service; its flag is passed to `serve.server`.
The upstream download links in the inherited documentation install upstream.
Use the branch-specific instructions above to try a fork contribution.

## Retained historical branches

These preserve earlier development and review snapshots. Their dated results
describe those snapshots; they are not tests of the current contributions.

| Branch | Documentation |
| --- | --- |
| [`feat/gfx1012-hip`](https://github.com/CC-David-CC/Strata-a5500/tree/feat/gfx1012-hip) | Earlier combined AMD and serving work: [manual build](https://github.com/CC-David-CC/Strata-a5500/blob/feat/gfx1012-hip/docs/AMD_HIP.md#rdna1-rx-5500-xt-8-gb-gfx1012) and [September 30 evidence](https://github.com/CC-David-CC/Strata-a5500/blob/feat/gfx1012-hip/docs/benchmarks/2026-09-30-gfx1012.json). |
| [`contrib/gfx1012-hip`](https://github.com/CC-David-CC/Strata-a5500/tree/contrib/gfx1012-hip) | Earlier AMD review branch, superseded by the separate hardware and serving contributions: [manual build](https://github.com/CC-David-CC/Strata-a5500/blob/contrib/gfx1012-hip/docs/AMD_HIP.md#rdna1-rx-5500-xt-8-gb-gfx1012) and [evidence](https://github.com/CC-David-CC/Strata-a5500/blob/contrib/gfx1012-hip/docs/benchmarks/2026-09-30-gfx1012.json). |
| [`contrib/p4-only`](https://github.com/CC-David-CC/Strata-a5500/tree/contrib/p4-only) | Earlier isolated P4 draft: [manual CUDA 12.x build](https://github.com/CC-David-CC/Strata-a5500/blob/contrib/p4-only/docs/DETAILS.md#experimental-tesla-p4-pascal-sm_61) and [evidence](https://github.com/CC-David-CC/Strata-a5500/blob/contrib/p4-only/docs/benchmarks/2026-09-30-sm61.json). Its README identifies measurements from the development revision, not a separate retest of this draft. |
| [`strata-p4`](https://github.com/CC-David-CC/Strata-a5500/tree/strata-p4), [`contrib/pascal-sm61`](https://github.com/CC-David-CC/Strata-a5500/tree/contrib/pascal-sm61) | Earlier P4 development snapshot, including the earlier AMD work: [manual build](https://github.com/CC-David-CC/Strata-a5500/blob/strata-p4/docs/DETAILS.md#experimental-tesla-p4-pascal-sm_61) and [evidence](https://github.com/CC-David-CC/Strata-a5500/blob/strata-p4/docs/benchmarks/2026-09-30-sm61.json). |

The historical P4 branches require `-DSTRATA_EXPERIMENTAL_SM61=ON` with
`-DCMAKE_CUDA_ARCHITECTURES=61` and CUDA 12.x. That branch-specific flag is
off by default. Current upstream instead has `STRATA_EXPERIMENTAL_SM60`;
do not mix instructions from different revisions.

## Evidence availability

Each linked measurement report has a JSON file committed beside it. These
files are publicly downloadable on the named branch; a local fleet directory
is not needed to read them. October 1 performance matrices retain their 0.1.33
source identities. October 2 rebase checks cover the stated 0.1.34 checks only.

Paths such as `${HOME}` and `/path/to/...` in recorded configurations must be
replaced for your installation. Full local logs and token arrays described as
locally retained are not additional public downloads. Some combined validation
commits were local; the reports give public source commits and merge recipes
to reproduce their trees. Model weights and generated binaries are obtained
or built separately, as described in the build instructions.
