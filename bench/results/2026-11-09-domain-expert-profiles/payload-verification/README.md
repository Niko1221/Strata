# Source GGUF payload verification

Author: **Francisco Angulo de Lafuente** · [ORCID](https://orcid.org/0009-0001-1634-7063)

Both complete source GGUF shards match the pinned Hugging Face LFS SHA-256 values:
**75,839,998,528 bytes across two files**. The [receipt](receipt.json) records the
expected and actual digests, sizes, timestamps and unchanged file statistics.

The check used a sequential full read with Python's SHA-256 implementation,
8 MiB chunks, CPU nice value 19 and idle I/O priority. It loaded no model and
used no GPU. The run completed on 2026-11-09 in 156.45 seconds.

[payload_check.py](payload_check.py) is the exact executed source. Its SHA-256 and
the input metadata SHA-256 are recorded in the receipt. The frozen
[source-model-metadata.json](source-model-metadata.json) is byte-identical to the
original [model-source.json](../model-source.json), captured before this full
payload check. Its initial metadata-only verification statement is historical.

## Reproduce

On Linux with Python 3 and `ionice`, copy this directory to a new scratch folder.
Keep the script and source metadata; remove the copied receipt in the scratch
folder before starting a fresh check. The model root must contain `IQ3_XXS/` with
the two named shards. Model weights are not included in this contribution.

```sh
python3 payload_check.py inspect --model-root /path/to/model
python3 payload_check.py start --model-root /path/to/model
python3 payload_check.py status
```

`start` returns a process handle. `status` checks that same PID and process start
time, or returns the final receipt. Repeated start calls do not launch another
worker. The worker has a 30-minute self-deadline. Success requires the receipt's
`status` to be `succeeded` and both `match` fields to be true; the status command's
exit code alone is not a verification result.

## Observations

This check identifies the bytes of the source GGUF files present at the recorded
time. It does not re-derive or attest the prepared native pack, reproduce the earlier
inference runs, measure answer quality or establish a speedup. All earlier frozen
performance and output-parity gates remain unchanged.
