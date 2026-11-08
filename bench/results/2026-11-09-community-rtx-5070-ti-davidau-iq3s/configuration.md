# Configuration provenance and portable commands

This is a transcription of observations in the experiment's local session notes, not a new server check. Paths below are placeholders. The publication preparation was performed offline; no server was contacted.

## Effective engine arguments observed in both versions

```text
<ENGINE-0.1.55-OR-0.1.58>/strata.exe --serve --pack <PREPARED-PACK> --native <MODEL>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf <MODEL>/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile <EXPERT-PROFILE>/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <MTP-RUNTIME> --max-context 131072 --kv int8 --batch 2
```

The loopback API port was 8089. Authentication was configured in the Python serving layer; the engine argument line alone is not the full API launch configuration. Credentials, private paths and access configuration are intentionally excluded. Vision was off. No retuning was performed. Prepared-component hashes were not captured.

## Measured benchmark commands (portable transcription)

```powershell
# The credential environment variable was set only on the inference machine.
strata bench run --url http://127.0.0.1:8089 --credential-env STRATA_API_KEY --suite standard-v1 --out <NEW-RESULT-DIRECTORY> --track custom --track-notes "DavidAU IQ3_S 128k int8 batch2 MTP; pre-update or post-update" --quantization IQ3_S --output json
# Community used STRATA_KEY as its credential environment reference.
python tools/strata_bench.py --url http://127.0.0.1:8089 --long --out <NEW-COMMUNITY-DIRECTORY>
```

Each command used that arm's installed release and tools. These are documentation examples with placeholders, not commands executed during publication preparation. Original run.command strings are retained with the private output path redacted. Community's generated command string omits the URL and output path; the transcription above adds them from the experiment notes.
