# Replay method (weights, harmonic mean, oracle check)

- Arm/task decode tok/s = mean over requests weighted by generated tokens (prompt 44/42/57 identify the task; requests with other prompt lengths are warm-ups and excluded).
- Warm = prefix-reused round (engine log "N reused"); cold = the first request of a task family.
- Acceptance = sum(drafts accepted) / sum(drafts offered), per arm/task over all that task's requests.
- Mixed workload score = harmonic mean of code and fiction warm decode tok/s (equal token counts).
- Oracle = per-task best arm (max warm decode), scored the same way. Compare against the best single static arm: if oracle == static, per-task switching has no headroom in this data.

Result on this data: MTP 48.4, DFlash2 static conf 0.70 48.8, oracle 48.8 — oracle ties static.
The warm-round-2-only code values (72-97 tok/s) that suggested conf 0.35 fastest on code are a selection artifact; token-weighted, conf 0.70 matches them with the highest acceptance.
