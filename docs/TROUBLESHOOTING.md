# Troubleshooting

This page is for anyone whose install or startup fails. Find your symptom below. Each entry gives one fix.
Start from the route you installed: [desktop app or Terminal](INSTALL.md), or [source checkout](SOURCE_INSTALL.md).
The normal release uses the app or `strata launch` and needs no `.bat` file. Detailed engine errors are in the
[engine reference](DETAILS.md#troubleshooting).

**Quick check.** Run these and keep the output for a bug report:

```sh
strata --version
strata doctor --live
strata launch --status
```

## The desktop app and the Terminal

**I cannot find `START-HERE.bat` or `SETUP.bat`.**
Open the installed Strata app, or run `strata launch`. Those wrappers are in the source repository and are not the
startup path for the app. For a fresh desktop install, download the one installer for your system
(`Strata-0.2.6-windows-x64-setup.exe`, `Strata-0.2.6-linux-x86_64.AppImage` or `Strata-0.2.6-macos-arm64.dmg`), not a
`strata-engine-…` archive.

**`strata` is not found.**
On Windows, open a new PowerShell window after installing. On Linux or Mac, use the full path the installer printed,
normally `~/.local/bin/strata`, and add `~/.local/bin` to PATH if needed. Check `strata --version`.

**Strata could not find an engine for this computer.**
Run `strata doctor --live` and `strata update --check`. Check the selected GPU, driver and
[backend guide](INSTALL.md#what-you-need). Supported paths include Apple Silicon and Intel Arc on Linux; do not
assume every machine without an NVIDIA/AMD discrete card is unsupported. A model download can finish even when
no engine starts. Verify `strata launch --status` and `/health` before calling the installation successful.

**This model needs a newer engine.**
Unload running models, then use the app's engine update or `strata update`. Minimum engine versions vary by model and
backend. Check the model guide and the installed engine, not only the app version. If the feed has no compatible engine,
report the missing combination.

**An update says it succeeded but `strata --version` is still old.**
The Terminal switches to the updated Control on the next command, so run `strata --version` again. If it is still old,
run the one-line installer once more; it keeps models and settings. The app, Control and engine have separate version
numbers; see [updating](INSTALL.md#updating) and the [release notes](https://github.com/Niko1221/Strata/releases).

**A new engine misbehaves after an update.**
Stop the model, then use the app's rollback control or `strata components rollback`. Source-checkout engines use a
separate rollback workflow, `python setup.py --rollback-engine`, described in [source setup](SOURCE_INSTALL.md).

**The Mac app cannot be opened.**
Check that the download is for Apple Silicon and that macOS is 15 or newer. Follow
[Mac installation](MAC.md#the-desktop-app-and-strata-launch-on-a-mac). A warning alone does not establish that a file
is safe or intact: verify the download source and release before changing quarantine settings. Release signing and
test-build handling are described in [Mac packaging](../packaging/mac/README.md).

**The Linux AppImage does not open.**
Make the downloaded file executable. It may need FUSE 2 (`libfuse2` on Ubuntu 22.04, `libfuse2t64` on newer Ubuntu;
package names vary by distribution). Without FUSE, run the file with `--appimage-extract-and-run`. For a crash log,
run it with `--save-logs`; see the logs section below.

## While installing or starting

**A download stopped.**
Retry it in the app, or rerun the same `strata launch` command. Completed model-download steps are reused. Check
free space and the error message. For an interrupted Terminal installer, rerun the install command. Only a source
checkout uses `START-HERE.bat` / `./setup.sh` to resume source setup.

**Startup takes a long time or the computer becomes slow.**
Large models need time to check files, prepare packs and fill memory. A first launch can take many minutes; there is
no fixed startup time. Watch progress and the engine log. If memory pressure is high, close memory-heavy programs and try a
smaller model or context. DeepSeek V4.1-Flash needs 46 GB of RAM at least ([model page](MODELS.md#deepseek-v41-flash));
`strata launch --plan --family deepseek41 --model ist-das-2.00bpw` shows what your computer is missing.

**The NVIDIA driver is too old.**
Use the driver required by the engine package. CUDA 13 needs driver 580 or newer; older cards use the
[CUDA 12 route](OLDER_GPUS.md). After updating the driver and restarting if needed, open the app or rerun
`strata launch`.

**Port 8080 is already in use, or I changed a running model's settings.**
Check `strata launch --status`. To replace your existing launch, rerun the intended command with `--restart`;
it stops current answers. To run beside it, choose another port, for example `--port 8081`, if memory permits.
Do not stop an unrelated program just because it holds the port. `--yes` alone does not restart a running model.

**The engine stops on a small GPU or low-memory computer.**
Try a smaller pack or context. The draft layer also uses memory. In the app, set the model's Draft layer to Off;
in the Terminal, an explicit example is:

```sh
strata launch --family qwen --model q2_0 --context 32k --mtp off
```

Use `--restart` if replacing an active run. Turning MTP off can slow decoding. Keep some VRAM free for the display;
setting the reserve to zero is not a general speed fix.

**Pictures do not work.**
The model, backend and installed engine package must include a compatible picture reader. Inspect the startup
error and the model guide. `--vision off` disables reading pictures; it is different from optional
[image generation](IMAGES.md). In the Terminal, use `on|off|gpu|cpu`, not source setup's `yes|no` spelling.

**Windows blocks an executable.**
Verify the official release and the component named in the message. Feed signatures and Windows code-signing are
different checks. If only the picture reader is blocked, `--vision off` may let text requests run. Do not disable
system-wide protection as the routine installation fix; include the blocked file and exact message in a bug report.

## While it answers

**It is slow and the disk stays busy.**
The chosen model may exceed available RAM/VRAM and be using the file tier. Close other programs, check memory in
the monitor, and choose a smaller model or context. Compare performance only with the same hardware and workload.

**The prompt exceeds the context.**
Start a new chat, shorten the input, or use the app's Summarize and continue action. A larger context needs more
memory. In the Terminal, restart your chosen model with an explicit `--context`, such as `64k`; preserve its family,
model and other choices. Do not look for `SETUP.bat` in an app install.

**On Linux the engine exits with code -9.**
Check system logs for an out-of-memory kill. Also check `ulimit -R`: a desktop-session real-time limit can cause a
kill even when memory is available. The engine prints a warning for that condition; see the
[detailed troubleshooting table](DETAILS.md#troubleshooting) before changing system limits.

**A hand-edited engine option disappears.**
The run config tracks explicit options in `user_args` and the generated baseline in `args_written`. Prefer the
app's Expert settings or Terminal `--set` for supported overrides. Context, KV, picture reading and drafting are
launch choices; set them through the app or the corresponding `strata launch` flags. Inspect the effective config
rather than assuming a saved preference was applied.

## Source-checkout problems

Python/venv, CUDA compiler versions, ROCm builds, `UPDATE.bat`, `update.sh` and generated `run-<model>` scripts
belong to [SOURCE_INSTALL.md](SOURCE_INSTALL.md). Use the backend guide for build prerequisites. The packaged
Terminal carries Python; a missing system Python is not a reason to install a compiler for the normal release.

If `git pull` fails in an old clone, back up your local work and follow the recovery steps the source update script prints.
Do not overwrite local branches or edits just to make `git pull` succeed.

## Logs and bug reports

- **App:** Settings → Engine → Diagnostics → View logs → Save logs. The diagnostic file goes to Downloads.
- **Terminal:** include `strata --version`, `strata doctor --live`, `strata launch --status`, and the relevant log excerpt.
- **Windows app crash:** inspect `%LOCALAPPDATA%\Strata\logs` for `desktop-crash-<date>.log`.
- **Linux app crash:** start the AppImage with `--save-logs` to collect diagnostics.

Include model and quantization, OS, GPU/VRAM, RAM, driver, install route, exact command, expected result and actual
error. Remove API keys and private data before sharing. Follow [CONTRIBUTING.md](../CONTRIBUTING.md).
