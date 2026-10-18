# Analysis method

The primary effect is the geometric mean of the paired A/U decode-token-throughput ratios. All 100 normal pairs are included, including early EOS. Each of the four workload/temperature cells has 25 pairs spread across 20 engine blocks.

Resample the 20 complete paired blocks with replacement 20,000 times using bootstrap seed 2026101802592. Use the same sampled block IDs for all cells. Report percentile 95% intervals with linear interpolation at `(n-1)*q`. The overall figure gives equal geometric weight to the four cell estimates.

Full-cap, same-output and order-stratified results are secondary summaries. Output lengths, slow samples and early stops remain in the raw data. Token throughput includes generated work; outputs can differ between versions.
