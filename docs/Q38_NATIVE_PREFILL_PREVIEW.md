# Qwen3.8-27B native prefill preview

This Windows preview adds faster short-prompt F16 prefill to the Strata
native CUDA 13.5 engine. It is an optional engine component for the
existing 0.6.3 desktop app, not a new installer or an automatic update.
The component reports version 0.6.3. It does not replace the existing
v0.6.3 release. The tested GPU was an RTX 5060 Ti with 16 GB VRAM.

[Preview downloads](https://github.com/zapabob/Strata/releases/tag/v0.6.4-preview.1) ·
[Base source](https://github.com/zapabob/Strata/commit/a4fec1973e45132a520519c4ce2322d1aff7fe11).
The release includes the four-file source patch used to build the native runner.
The tag points to the equivalent source snapshot (`c463ef1cc0dd`).

## Try it

Finish requests, unload any model, and quit Strata, including its tray
icon. Keep your existing engine available for rollback. Download the
preview engine archive and its separately published `feed-key.pub.json`.
Check both files against the SHA-256 pins below before trusting the key.
Keep the public-key file in a local directory you control, and use its
absolute path in this PowerShell example:

```powershell
$env:STRATA_UPDATE_TRUSTED_KEYS = 'C:\path\to\feed-key.pub.json'
$env:STRATA_Q38_SHORT_F16_BATCH = '1'
& "$env:LOCALAPPDATA\Programs\Strata\strata-desktop.exe"
```

These settings apply only to this PowerShell session and child
processes. The public key adds trust for the preview; it does not
disable signature verification. Import the archive in the app using
**I already have the file**, select it, and load the supported
[3.69 bpw mixed-quantized GGUF by soyaakinohara](https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf)
with a 4,096-token context and F16 KV. The model is a separate download.
The fast path applies only to eligible short prefill batches; larger
or unsupported calls use the existing path. Thanks to **soyaakinohara**
for the GGUF and **AEON-7** for the source weights.

To return to the earlier engine, unload the model and select the
previous component in the app. Quit the preview session and start
Strata normally without these environment settings.

## Verification and limits

The signed payload passed installed-GUI Japanese chat, mid-response
Stop, explicit unload, and clean process exit. Both tested turns
used the new prefill path. The isolated test added the public key
and enabled the option explicitly, as described above.

The exact payload shortened initial native prefill by 35–43% when
enabled versus disabled in the fixed-token test. In two initial-request
comparisons to Prism llama.cpp with the same GGUF and F16 KV, native
prefill took 743–755 ms versus 756–822 ms. Native decode took 3–12%
longer. A separate warm benchmark build, with the same prefill
implementation and a repeat-only CLI loop, remained behind Prism in
both orders: native prefill 312–388 ms versus 208–211 ms, and native
decode 27.10–27.33 versus 30.52–30.91 steps/s. These tests do not
establish full-window performance or general model quality. The shipped
desktop installer does not have a Windows Authenticode signature.

| Asset | SHA-256 |
|---|---|
| `strata-engine-0.6.3-windows-x86_64-cuda-cuda13.tar.gz` | `53bfa31fc0f6be23394c40f05e7101834d7f6b70598217252571ec2753f07b9f` |
| `feed-key.pub.json` (public key ID `9c835e4839a4c6f3`) | `48c36a225cd4ef82fd04cc8667a38230698fa9e29d1a4b187a963393e32f6ef1` |
| `q38-short-prefill-source.patch` | `6d26014144a7296defe59363de51dac2326937520e0482107b3f4626ba72acc8` |
