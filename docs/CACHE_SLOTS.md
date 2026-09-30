# Independent conversation cache slots

An application can alternate a long foreground conversation with short background requests (summaries, memory extraction, reviews). A single positional KV arena otherwise loses the foreground branch when the background prompt diverges. Prefix checkpoints alone cannot restore overwritten positional cells.

For single-GPU sessions, send `strata_cache_slot` with an integer from 0 to 3 on `/v1/chat/completions` or `/v1/messages`. Omission selects 0, preserving the existing single-cache behavior. Assign stable slots in the client; for example, chat=0, extraction=1, background work=2, review=3.

```json
{
  "model": "your-loaded-model",
  "messages": [{"role": "user", "content": "Continue the current task."}],
  "strata_cache_slot": 0,
  "max_tokens": 256
}
```

This is cache isolation, not parallel generation. Requests still use the existing FIFO and one active session arena. Slot IDs belong to the server, not to an authenticated client; clients sharing an ID share a cache. Exact token and image-prefix checks still decide whether a saved prefix is reusable. A slot does not supply messages omitted from the next request.

## What is saved

On a slot switch, the outgoing cache keeps its live token/image identity, recurrent GDN/PLE/index-tail state, prefix checkpoints and control-vector setting. Main and MTP positional KV cells and pooled index data are copied to a temporary file in bounded chunks. The recurrent state, token/image identity and prefix checkpoint payloads are appended to the same file, then their RAM buffers are released. Only small size descriptors remain in RAM for inactive slots. Restoring a slot writes into the existing arena and resets streamed GPU page mappings; weights and captured graph addresses stay resident. Cancelled requests can save their latest valid checkpoint instead of an incomplete live state. The existing pinned-root/LRU checkpoint policy remains in effect within each slot.

There can be three inactive files at once. All inactive cache payloads use temporary disk space. The active slot still retains its recurrent state and prefix checkpoints in RAM; its memory usage depends on `--prompt-cache`. The engine logs the checkpoint MiB offloaded and total switch time. OS filesystem caching can temporarily use reclaimable RAM for these files; offloading does not guarantee an identical increase in free RAM. This is not four copies of the model or four GPU arenas. Files are delete-on-close and are not durable conversation storage. An engine restart clears all slots. File/CUDA snapshot failures are reported as engine errors rather than returning an answer from partial state.

Layer-split multi-GPU sessions advertise one slot and reject nonzero selections. Their multiple arenas would need a separate implementation; slot 0 keeps their previous behavior. The other single-GPU KV formats share the snapshot code, but the live validation below covers int8 KV on NVIDIA only. HIP, image-cache rotation and other KV formats have not been live-validated.

## Verification

Local SSD-offload trial (Windows, RTX 4070 Ti, WD_BLACK SN850X, Qwen IQ2_XS, int8 KV, 32k resident window): the 35k-token rotation/retrieval/cancellation check passed with identical cold/restored answers. Each long inactive slot offloaded about 450.8 MiB of checkpoint state. Peak private-memory growth above the first cold request was 899.9 MiB before versus 34.2 MiB after offloading, a reduction of 865.7 MiB in this workload. Populated-slot switches took 437–938 ms total. These are one-run local measurements, not a general latency guarantee; filesystem caching, checkpoint counts and storage pressure affect results. The installed Little Bot Responses relay also completed a streamed response against the new engine.

Run server tests with `python -m unittest discover -s serve`. The slot tests cover default/range/type validation, forwarding on both APIs, rejection before SSE headers, and the layer-split restriction.

For real inference, close other clients, load a model and run:

```
python tools/check_cache_slots.py --url http://127.0.0.1:8080 --lines 1700
```

The script replaces all four slots with synthetic prompts. It requires zero cached tokens for independent cold requests, reuse after rotation, identical greedy cold/restored answers, successful retrieval of two distant records, and parity after disconnecting a streaming generation. It discovers the loaded model from `/health` and prints timings and cached-token counts; it does not infer reuse from latency. For reproducible parity, use `--adapt-swaps 0` to keep expert placement fixed. With Qwen the default prompt is about 35k tokens, beyond a 32,768-token resident KV window. Existing cached-token usage fields are unchanged.
