# Galahad session storage

Strata can use [Galahad](https://github.com/corbenicai/galahad) to store complete named conversation states through
its existing `POST /slots/0?action=save|restore` API. This is optional. Without the setting below, Strata uses the
same engine and file session path as before and does not import Galahad.

This adapter stores the entire Strata session image, including attention KV, recurrent state, checkpoints and
draft state. Galahad manages the durable store and encryption; Strata still validates the model, configuration
and session format before restoring the image. A Galahad block is not a llama.cpp session file.

This version does not automatically save requests or search for shared token prefixes, retrieve documents with
Blaise, or add Galahad branching. Save and restore are explicit client operations. It supports the single slot 0
and cannot be combined with parallel requests or `slot_save_path`.

## Install and enable

[Galahad's installation guide](https://github.com/corbenicai/galahad/wiki/Install) lists Linux x86-64, Python
3.10–3.14 and a licensed NVIDIA GPU as requirements for the current package. The adapter uses the Python bindings
shipped in `galahad-kv` 1.31.5 (C ABI 1.31). Windows and AMD are not supported by that package. Install it into the
same environment that runs Strata:

```bash
.venv/bin/python -m pip install galahad-kv==1.31.5
.venv/bin/galahad doctor
```

Install your licence using Galahad's guide before enabling storage. Licence activation and acceptance of its terms
are separate from Strata setup. Set `GALAHAD_LICENCE_FILE` if the licence is not in its default location. Galahad
reads its key material from your configured licence/key provider; Strata does not generate or log those keys.

Add this object to the existing `strata-*.json` config:

```json
"galahad": {
  "cache_dir": "sessions-galahad",
  "model_fingerprint": 123456789,
  "max_session_mib": 2048
}
```

`123456789` is an example: replace it with a positive 64-bit identity for your exact model, tokenizer and
quantisation. Use a new identity when those change. Do not share the directory with a different engine or model.
Strata's own session identity checks also apply on restore. A relative `cache_dir` is relative to the config's
`cwd`, or the server's working directory when `cwd` is absent. Remove `slot_save_path` if it is configured.

Alternatively, pass `--galahad-cache-dir DIR --galahad-model-fingerprint INTEGER` to `serve/server.py` together
with the existing `--engine strata --config ...` arguments. The command-line directory is relative to the server's
working directory. The maximum image size defaults to 2048 MiB; change `max_session_mib` in the config if needed.

## Save and restore

After a completed chat request, save the conversation the engine currently holds:

```bash
curl -X POST 'http://127.0.0.1:8095/slots/0?action=save' \
  -H 'Content-Type: application/json' -d '{"filename":"chat-001.bin"}'
# Later, including after restarting the same compatible engine:
curl -X POST 'http://127.0.0.1:8095/slots/0?action=restore' \
  -H 'Content-Type: application/json' -d '{"filename":"chat-001.bin"}'
```

Add `Authorization: Bearer YOUR_KEY` if the server requires an API key. The usual Host, Origin and filename checks
apply. Keep the server bound to `127.0.0.1` unless an API key is configured.

The response fields match the file session API: `id_slot`, `filename`, `n_saved`/`n_restored`,
`n_written`/`n_read`, and `timings.save_ms`/`restore_ms`. Timings include the engine operation and Galahad transfer.
Names are immutable on this backend: saving a name already present returns `409`; use a new name for each version.
Restore of a missing name returns `404`. An empty or oversized image returns `413`; Galahad's disk budget refusal
returns `507`. Other storage failures return `500`. A failed or partial load never reaches the engine. A storage
failure does not switch silently to the ordinary file backend.

The adapter uses confirmed lookups with two hashes, checkpoints after a save and rehydrates the store at startup.
It stages the plaintext image in a private temporary directory because the engine accepts file paths. The
temporary image is removed on success and handled errors; an abrupt process kill or power loss can leave it in the
OS temporary directory. Provision temporary disk space for one full image as well as Galahad's store. Memory maps
avoid an additional Python byte-string copy; this is not a zero-copy GPU transfer. The image is opaque to Galahad,
so no homogeneous floating-point dtype is declared for its value scanner; Strata performs session validation.

Save captures the engine state when the operation gets the service lock. Clients that need a specific request's
state must arrange that no other chat request runs between that request and save.

## Validation and limits

```bash
.venv/bin/python -m unittest serve.test_galahad serve.test_slots serve.test_security
```

These tests use a fake Galahad library and engine, covering byte-exact adapter round trips, reinitialisation,
confirmation mismatches, missing/duplicate names, size bounds, incomplete loads, storage errors, cleanup, and the
unchanged file backend. They do not establish native Galahad persistence, encryption or GPU output parity.
The native library from the 1.31.5 wheel was loaded on Linux and reported ABI 1.31; initialisation without a licence
and key material refused with `GLH-E17`. A licensed native round trip was not tested.
On a licensed machine, save an actual conversation, restart Strata, restore it, and continue with the same history.
Compare it with the ordinary session file path on the same model and build. No Strata performance measurements
with Galahad are published by this change.
