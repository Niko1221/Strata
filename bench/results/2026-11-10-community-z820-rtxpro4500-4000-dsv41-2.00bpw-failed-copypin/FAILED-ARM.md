# INVALID ARM — do not use these numbers

This folder is the first attempt, run with `DS4_COPY_PIN=1`. Chat worked
(median 28.7 tok/s), but every 4K / 16K / agent / 32K prompt failed with:

    cudaMemcpyAsync(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2),
                    cudaMemcpyHostToDevice, c.cs): invalid argument

The forced copy-pin made the prompt path DMA the pinned RAM copy; the clean arm
removed `DS4_COPY_PIN` and completed all stages. Keep this folder only as
evidence of the failure; the performance report is
`../2026-11-10-community-z820-rtxpro4500-4000-dsv41-2.00bpw-fill-6exp-0995-0512/`.
