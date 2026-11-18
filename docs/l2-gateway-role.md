# L2 gateway as a managed role of the Control process group

## Position

The gateway stays a separate process (branch `l2-gateway-reference`), but it
can optionally register itself into the Control process group instead of
being a foreign bystander:

```
STRATA_GW_ROLE=1 python l2/gateway2.py
```

Registration reuses `strata_control/procs.py` directly (same repo, same
protocol): a `{state}/procs/{pid}.json` record with
`role="gateway"`, the start token, the command line, and the parent token.
Unset, the gateway is exactly the standalone proxy of before.

## What this buys, with zero Control-side changes today

- `leftovers()` / `end_leftovers()` already sweep every record in the procs
  directory: a gateway left behind by a crashed Control is ended (grace 8 s)
  by the next Control start, the same as a stray engine.
- `records()`-based tooling sees the gateway next to the engine.

## Control-side follow-up (after #5500 merges)

Two small hooks, both behind the role name:

1. a launch hook in the serve startup path: when `STRATA_GW_ROLE` is set on
   the Control side, start the gateway from the repo tree and hand it the
   engine URL (one `popen(role="gateway")` call);
2. a health联动 in `management.py`: the gateway's `/__gw/status` joins the
   set of endpoints the web page's process table renders.

Neither hook changes any default: without the role, nothing new runs.

## Failure semantics

- Registration is best-effort and never fatal (missing strata_control on
  PATH, unwritable state dir, no token: the gateway logs one line and runs
  unmanaged).
- The record is dropped via `atexit` with the start-token guard, so a PID
  reuse cannot delete a later process's record.
