# Strata 0.3.0-preview.20261024.1: unfinished native CLI

This archive contains a native Windows x64 CUDA engine and its required DLLs.
It is not a GUI installer, free-text chatbot or HTTP server. Keep all DLLs next
to strata.exe. No Node, Python or CUDA Toolkit installation is required for
the executable. Windows system libraries and an NVIDIA driver are required.

Tested machine: Windows 11 Pro 10.0.26300, Ryzen 5 4500, 32 GiB RAM,
RTX 5060 Ti 16 GiB (sm120), NVIDIA driver 616.92. An older minimum driver
has not been qualified. STRATA_PORTABLE is OFF and the MSVC runtime is
dynamic; compatibility with other Windows CPUs has not been established.
The engine reports 0.1.47. The distribution has its own preview number.

In PowerShell from this extracted directory:

```powershell
.\strata.exe --help
```

For the bounded diagnostic, provide an existing matching Qwen3.8-Flash-Next
GSQ-RCO Q2_0 native pack and its original two shards in this relative layout.
The archive contains no weights, model pack or download tool. The token file
is specific to that Qwen tokenizer. Do not use it with Bonsai or another model.

```powershell
.\strata.exe --pack .\models\q2_0\native-pack `
  --native .\models\q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf `
  --ple-gguf .\models\q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf `
  --tokens-file .\examples\ready-tokens.txt --max-new 32 --max-context 256 `
  --mmap-experts --expert-cache auto --expert-profile .\data\expert-profile.bin `
  --prefill 32 --eos-ids 248044,248046 --spin8-mode off --check-logits --gpu 0 `
  --spec 2 --suffix-draft 0 --lookup-chain 0 --vram-reserve-mib 6144
```

The fixed 30-token prompt asks for READY. The final source build produced
READY and EOS in off mode. This diagnostic used source validation guards,
fresh resource admission and no concurrent model/compiler job. The direct
command above does not install those guards. These settings are a recorded
diagnostic, not a universal memory recommendation. Prepare packs and other
prompts with the separate repository tools and the selected pack's tokenizer.

F32 three-view admission and prefill passed, but generation failed after
automatic expert-cache sizing selected zero slots with a 9216 MiB reserve.
Compressed three-view model generation was not run. Keep experimental modes
off. They require manually validated capsules and source tooling; no local
machine capsule is included. Bonsai, full model quality, 4K/32K contexts,
speed improvements, GUI/HTTP, persistence, monitoring and other backends are
not qualified. This is not a completed 0.3.0 release.

The manifest records payload hashes and the build/source identities. The
release SHA-256 covers the entire archive. Hashes detect corruption; they
do not provide an identity signature. The executable is unsigned. See the
separate qualification.json release asset for checks on the final extracted
archive, including the distinction between startup and actual inference.

Strata and ggml retain MIT terms. The bundled NVIDIA and Microsoft runtimes
retain their respective proprietary licence terms, supplied unmodified under
licenses/. They are not covered by Strata's MIT licence. By using this bundle
you agree to those applicable component terms. Distributors must preserve
these terms and notices and require downstream users and distributors to
agree to terms protecting the vendor components at least as much as their
supplied agreements. Do not modify vendor binaries or distribute them as a
standalone offering. See THIRD-PARTY-NOTICES.md for origins.

Source, detailed limitations and research context:
https://github.com/zapabob/Strata/blob/main/docs/releases/preview030.md

Download:
https://github.com/zapabob/Strata/releases/tag/v0.3.0-preview.20261024.1
