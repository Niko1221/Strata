# Idle session offload

For a single CUDA GPU, the serving engine can save its session state to disk
while keeping model weights loaded. This releases the session's physical VRAM
and host checkpoint payloads between requests. It is disabled by default.

Add these engine arguments:

```
--cache-idle-seconds 180 --cache-expire-seconds 3600 --cache-disk-dir /path/to/cache
```

Both intervals are measured from the end of the last generation. Expiry must
exceed the idle interval. The directory needs room for the whole allocated
session, including unused context capacity. CUDA virtual memory support and
fully resident KV are required; batch, pipeline, multi-GPU, remote experts and
elastic KV are rejected when this option is enabled.

The engine saves between requests after pending GPU work completes. It flushes
the snapshot before releasing memory. A failed write retains the in-memory
cache and retries later. Restoring remaps the same virtual addresses so captured
graphs remain valid. Missing, expired or corrupt snapshots discard the cached
conversation; the next generation reads its prompt again. Failure to remap GPU
memory terminates the engine. A save already in progress delays a new request.

Snapshots are temporary, process-local caches, not conversation exports. They
contain conversation state and should be stored in a private directory. Normal
shutdown removes them; startup removes expired files belonging to this feature.
`/v1/status` exposes `cache.state`, disk size, released bytes and operation time.

Validation on Windows, RTX 2080 Ti, CUDA, IQ3_XXS and INT8 KV at 4096 context:
the real HTTP lifecycle covered save/restore with a cache hit, corruption and
expiry with prompt reread, and cancellation followed by another generation.
The snapshot was 309,197,048 bytes; one measured save took about 5.3 seconds and
restore about 367 ms. These are local measurements, not throughput claims.
The native `idle_cache_test` checks exact multi-region restores and graph replay
after remapping, constants, invalid snapshots and cleanup. Other GPU backends
and hardware have not been runtime-tested.
