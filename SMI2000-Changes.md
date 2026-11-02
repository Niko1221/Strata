# SMI2000 changes: Monitor extras (memory, experts, sampling, KV, VRAM, requests)

Local change, not part of the official Strata release. This file says how to apply it again after every official
update. An LLM can follow it step by step: read "What it does", apply the hooks (section "Hooks"), add the new files
(appendix B), run "Check" and "Install".

Last applied to: Strata **0.2.9** (engine 0.1.55), commit `c62d5e3a`.

## What it does

All on the **Monitor** screen. The engine (C++) does not change. Only the Python server (`serve/server.py`), the host
(`apps/host/server`) and the web UI (`apps/host/src`) change.

1. **Memory and experts** card (Spanish UI: *Memoria y expertos*), under the energy and temperature row:
   - **System RAM**: used / total GB, a bar, % used, GB free, GB of experts the engine keeps in RAM, a 60 s chart.
   - **Expert placement**: one stacked bar and three figures, VRAM / RAM / NVMe disk (experts, GB, share).
2. **Clear** button (*Borrar*) in **Recent requests**. The first click asks ("Click again to clear"), the second
   clears the request list and the totals under it. The model keeps running. Only the `/metrics` view is reset:
   `/v1/status` and the Prometheus counters keep counting from the server's start.
3. **Sampling** card (*Muestreo*), left of "Memory and experts": temperature, top-p, top-k, min-p, repetition,
   presence and frequency penalty, penalty window, seed. These are the values a request gets for the fields it leaves
   out (model card, then the model's `sampling` config block, then the Chat settings). A dot marks the values set in
   the model's config. Two columns (Thinking / No thinking) only when they differ.
4. **KV cache** in **Context fill**: the `--kv` format (`int8`, `q4_0`, `k8v4`, `fp16`, DeepSeek `fp4`), its
   estimated size, and a short text: what the format is, what it costs or saves, KB per context token, and whether
   the KV streams from RAM (`--kv-resident`).
5. **VRAM bar** in the **VRAM** card: one coloured bar plus a legend in GB: model weights, MTP draft, expert cache,
   KV cache (≈), verify buffers, other (work memory, CUDA, other programs, picture reader) and free. Only when one GPU
   is shown and the model runs on one GPU.
6. **Prefill tok/s** column in **Recent requests**, right after **Tok/s**.
7. **Copy to clipboard** button (*Copiar al portapapeles*) in the **Monitor screenshot** window, between **Close**
   and **Download**: it puts the PNG on the clipboard (`navigator.clipboard.write` with a `ClipboardItem`) and says
   so. Checked in the preview: the clipboard then holds `image/png`.

Measured on this PC (RTX 3060 12 GB, 48 GB RAM, Qwen3.8-Flash-Next UD-Q6_K_XL, 64K, `--kv k8v4`,
`--resident-budget-gib 24`): experts VRAM 609 / 2.5 GB, RAM ≈5,796 / 24.0 GB (the engine log says 5,791),
NVMe ≈18,171 / 75.2 GB, of 24,576 / 101.7 GB. VRAM: weights 4.38 GB, MTP draft 1.13 GB, experts 2.51 GB,
KV ≈0.80 GB, verify buffers 0.11 GB.

## Where the numbers come from

| Figure | Source |
|---|---|
| Experts in VRAM | engine `INFO`: `expert_slots`, `expert_cache_mib` (DeepSeek: `expert_slots`) |
| Experts in RAM | engine `INFO`: `arena_mib` (Qwen, so the count is `≈`); DeepSeek `ram_experts` |
| Experts total | **new** in server.py: `experts`, `expert_total_mib` from the pack's `native_experts.txt` header (`n_expert`, `total`) × layer lines; DeepSeek `expert_slots + cpu_experts` |
| Experts on disk | total − VRAM − RAM (the OS file cache may hold some: not measured, the card says so) |
| Sampling | **new** `/metrics` key `sampling`: `{config, thinking, instruct}` = the server's `sampling_defaults`, and `model_card_sampling(arch, thinking) + sampling_defaults + shared` |
| KV format | engine `INFO`: `kv`, `context`, `kv_resident` |
| KV size | **new** `/metrics` `engine.kv_mib_est` = 121 + context × 13 × bytes(kv) × 1.054 / 2^20 MiB, bytes = int8 1056, q4_0 576, k8v4 816, fp16 2112 (the same figures as `setup.UD_FIT_KV_BYTES` and `strata_control/vram_fit.py`). Only for `arch` `qwen4exp` (Flash-Next: 12 QSA layers + the draft layer); none for other families |
| VRAM parts | **new** `engine.vram_parts` `{weights, draft, buffers}` (MiB), read from this engine start's log lines: "MiB of weights loaded from", "native projection matrices, N MiB of weights", "N GiB of Q8_0 hyper-connection projections" (weights); "draft layer loaded, N MiB of VRAM", "draft head over N tokens (N MiB)" (draft); "window up to N tokens, N MiB of device buffers" (buffers). A line that is missing leaves its part out |
| VRAM other / free | sensor used − known parts / total − used (parts above the reading are scaled down; the bar never passes 100%) |
| Prefill tok/s | already in each request: (prompt − reused) / prompt_ms (`promptTps`) |

Rules kept: nothing is guessed when a figure is missing (the card or part is left out or says so), and every
estimate is marked `≈`.

## Files

New files (all owned by this change, full content in appendix B):

| File | What |
|---|---|
| `apps/host/src/lib/smi-expert-tiers.ts` | pure helpers: `expertTiers()`, `kvDetails()`, `vramSegments()` |
| `apps/host/src/components/SmiMemoryPanel.tsx` | the Memory and experts card |
| `apps/host/src/components/SmiMonitorExtras.tsx` | `SmiClearRequests`, `SmiSamplingCard`, `SmiKvInfo`, `SmiVramBar` |
| `apps/host/src/styles/smi-monitor.css` | their styles |
| `apps/host/src/locales/smi-en.json`, `smi-es.json` | their strings (other languages fall back to English) |
| `apps/host/tests/smi-monitor.test.mjs` | UI helper, mapping and host tests |
| `serve/test_smi_monitor.py` | server tests |

## Hooks in official files

Each hook is small and marked `SMI2000` where a comment fits. The exact diff against 0.2.9 is in appendix A; apply it
with `git apply` when it still fits, or by hand from the descriptions below.

### `serve/server.py`

1. Before `def read_engine_stage_log(`: add `pack_expert_totals(args)`, `VRAM_LOG_PARTS`,
   `vram_parts_from_log(text)`, `SMI_KV_BYTES`, `SMI_KV_LAYERS` and `kv_estimate_mib(arch, kv, context)`.
2. `StrataEngine.__init__`, after READY: right after the `self.info.update(drafting.engine_info(...))` try/except and
   before `self.ended = False`, set `self.info.setdefault(k, v)` for each pack total, and
   `self.info["vram_parts"] = vram_parts_from_log(start_log_text(log, log_start))` in a try/except.
3. `Service`: a new method `reset_metrics()` just before `def metrics(`: clears `self.history` and stores
   `self.metrics_base` (the numeric totals now, and `since`), under `self.status_lock`.
4. `Service.metrics()`: after `totals = dict(self.totals)` (still under the lock), subtract `metrics_base` from the
   numeric totals (and use its `since`); after `engine = {...}`, set `engine["kv_mib_est"]`; at the end of the
   returned dict, add `"sampling": {"config", "thinking", "instruct"}`.
5. Handler `_post()`: after the `/config` branch, add `/metrics/reset`: `self._own_page(...)` (JSON only, no
   foreign Origin), then `svc.reset_metrics()` and `{"status": "cleared"}`. The API key check at the start of
   `_post()` already applies.

### `apps/host/server/native-control.mjs`

Before the factory's final `return {state:publicState,...}`, add `async function resetMetrics(id)`: it resolves the
instance, POSTs `http://127.0.0.1:<port>/metrics/reset` with `authorization: Bearer <instance.key>`, reads
`/metrics?requests=all` again into `instance.metrics` (keeping its `mcp`), `emit()`s and returns `publicState()`.
Add `resetMetrics` to the returned object. (The host's 2 s poll keeps older requests it already had; replacing
`instance.metrics` after the reset is what drops them.)

### `apps/host/server/index.mjs`

After the `nativeInstance` (`unload|default`) route: `POST /api/native/instances/<id>/metrics/reset` →
`native.resetMetrics(id)`, a 409 with the message on error.

### `apps/host/server/monitor-dashboard.mjs` and `apps/host/src/lib/native-monitor.ts`

In the `expertCache:{...}` object: `total` falls back to `expert_slots + cpu_experts`, and three new fields
`totalGb` (`expert_total_mib`), `ramGb` (`arena_mib`), `ramUsed` (`ram_experts`). After it, two new keys:
`engineInfo` (the `/metrics` `engine` object; in native-monitor only for the selected model) and `sampling`
(`/metrics` `sampling`).

### `apps/host/src/lib/types.ts`

`MonitorState.expertCache` gets `totalGb?`, `ramGb?`, `ramUsed?`; `MonitorState` gets `engineInfo?` and `sampling?`.

### `apps/host/src/screens/Monitor.tsx`

1. Imports: `SmiMemoryPanel` and `SmiClearRequests, SmiKvInfo, SmiSamplingCard, SmiVramBar`.
2. VRAM `MonitorMetric`: `chart={<>{the old chart}{one GPU ? <SmiVramBar engine={data.engineInfo} usedGb={vram} totalGb={vramTotal}/> : null}</>}`
   where "one GPU" is `selectedGpus.length === 1 && (selectedInstance?.gpuIds.length ?? 0) <= 1`.
3. After the readings strip: `<div className="smi-row" data-single={!data.engineInfo}>{data.engineInfo ? <SmiSamplingCard sampling={data.sampling}/> : null}<SmiMemoryPanel data={data}/></div>`.
4. Context fill: `<SmiKvInfo engine={data.engineInfo}/>` right after the gauge's `</div>`.
5. Recent requests toolbar: wrap the buttons in `<span className="smi-toolbar-actions">` with
   `<SmiClearRequests ids={native ? (selectedInstance ? [selectedInstance.id] : runtime?.instances.map(i => i.id) ?? []) : []}/>`
   before "Show all".
6. Table: `<th className="num">{tr("Prefill tok/s")}</th>` after the Tok/s header,
   `<td className="num">{tr(number(r.promptTps, 0))}</td>` after the Tok/s cell, and the empty row's `colSpan`
   one more (`10 : 9`).

### `apps/host/src/components/SnapshotDialog.tsx`

Add `copySnapshotWithNotice(data)` (base64 PNG to a `Blob`, `navigator.clipboard.write([new ClipboardItem({'image/png': blob})])`,
a success or error toast). In the dialog's `actions`, between the Close and Download buttons:
`<Btn icon="copy" disabled={!data || pending} onClick={() => void copySnapshotWithNotice(data)}>{tr("Copy to clipboard")}</Btn>`.

### `apps/host/src/main.tsx`, `apps/host/src/lib/i18n.ts`, `apps/host/tests/locale-gaps.test.mjs`

- `main.tsx`: `import './styles/smi-monitor.css';` after `monitor.css`.
- `i18n.ts`: import `smi-en.json` and `smi-es.json`; add `...smi_en` at the end of `english` and `...smi_es` at the
  end of the Spanish entry.
- `locale-gaps.test.mjs`: add `'smi-en'` to the English catalog list of the first test.

Rule for the catalogs: never put an English string in `smi-en.json` that already exists (same case) in `en.json`,
`tune-en.json`, `tools-en.json`, `fix-en.json` or `gh-en.json`: the last key wins and the other seven languages
would lose that translation. Check:

```sh
cd apps/host/src/locales
node -e "const f=require('fs');const k=new Map();for(const n of ['en','tune-en','tools-en','fix-en','gh-en'])for(const [i,v] of Object.entries(JSON.parse(f.readFileSync(n+'.json','utf8'))))k.set(v,i);for(const v of Object.values(JSON.parse(f.readFileSync('smi-en.json','utf8'))))if(k.has(v))console.log('DUPLICATE',v,k.get(v))"
```

On Windows, write these JSON files with a UTF-8 editor or a script file. Text piped through PowerShell into
`node -` turns `…`, `●` and accents into `?`.

## If the official code moved

- Find the spots with `grep -n "expertCache" apps/host/src apps/host/server`,
  `grep -n "monitor-readings\|monitor-context-requests\|Recent requests\|Tok/s" apps/host/src/screens/Monitor.tsx`,
  `grep -n "def metrics\|def read_engine_stage_log\|path == \"/config\"\|drafting.engine_info" serve/server.py`.
- If the engine starts to report `experts`, `expert_total_mib` or its VRAM parts itself, drop that part of the
  server hook (`setdefault` already lets the engine's numbers win).
- If the engine's start lines change their wording, fix `VRAM_LOG_PARTS` (the test in `serve/test_smi_monitor.py`
  holds the lines as 0.1.55 writes them).
- If the official Monitor gets its own version of one of these features, drop ours.
- If `native_experts.txt` changes its header, fix the two regular expressions in `pack_expert_totals`.

## Check

```sh
cd apps/host
npm ci
node --test tests/smi-monitor.test.mjs tests/locale-gaps.test.mjs tests/native-monitor-projection.test.mjs tests/monitor-host-sensors.test.mjs
npx tsc --noEmit
npm run build
npm test
cd ../..
python -m unittest serve.test_smi_monitor serve.test_vram serve.test_server
```

On 2026-11-02 (Node 22.4.1, Windows, the app's Python `C:\LLM\Strata\control\python\python.exe`): the 4 focused
files passed (25 tests), `tsc` gave no errors, the build passed; `serve.test_smi_monitor` and `serve.test_vram`
passed (13), `serve.test_server` passed (359, 3 skipped). Full `npm test`: the same 4 failures as without this change
(native file actions, engine-version, `monitor-dashboard.test.mjs` which cannot load `.ts` on Node 22.4, weight
discovery with symlinks); `timeskip-real` failed once in a full run and passed alone.

The UI was checked in the isolated preview host (`scripts/preview-ui.mjs`, with this PC's figures put in the
fixture's metrics): all six features show, and Clear empties the list.

## Install into the desktop app (C:\LLM\Strata)

The installed app runs the built web UI from `C:\LLM\Strata\host\dist`, the host from `C:\LLM\Strata\host\server`
and the Python server from `C:\LLM\Strata\control\serve`. An official update overwrites them, so apply this again
after each update. Your data and settings (`%USERPROFILE%\.strata`, `%LOCALAPPDATA%\Strata`) are not touched.

1. Check that the installed files match the checkout's version before you copy (they did for 0.2.9): each of the
   five files below (`SnapshotDialog.tsx` goes in through `dist`) must equal `git show HEAD:<path>`. If not, check out the tag of the installed version first.
2. Back up (PowerShell, once per version):

   ```powershell
   $bk='C:\LLM\Strata\.smi2000-backup'; New-Item -ItemType Directory $bk -Force | Out-Null
   Copy-Item C:\LLM\Strata\host\dist "$bk\dist" -Recurse
   Copy-Item C:\LLM\Strata\control\serve\server.py "$bk\server.py"
   foreach ($n in 'monitor-dashboard.mjs','native-control.mjs','index.mjs') { Copy-Item "C:\LLM\Strata\host\server\$n" "$bk\$n" }
   ```

3. Copy (from the repo root, after `npm run build` in `apps/host`):

   ```powershell
   Copy-Item apps\host\dist\* C:\LLM\Strata\host\dist -Recurse -Force
   Copy-Item serve\server.py C:\LLM\Strata\control\serve\server.py -Force
   foreach ($n in 'monitor-dashboard.mjs','native-control.mjs','index.mjs') { Copy-Item "apps\host\server\$n" "C:\LLM\Strata\host\server\$n" -Force }
   ```

4. **Close Strata completely and open it again** (the host is a Node process: its new route needs a restart), then
   load the model. The server reads the pack totals, the VRAM parts and the sampling when the model starts.
5. Check: `GET http://127.0.0.1:<port>/metrics` (port and `api_key` in
   `%USERPROFILE%\.strata\data\native-instances\<id>.json`) has `engine.experts`, `engine.expert_total_mib`,
   `engine.vram_parts`, `engine.kv_mib_est` and `sampling`.

To undo: copy the backed-up items back and restart Strata.

## Appendix A: the diff of the official files (against 0.2.9)

Apply from the repo root with `git apply --3way SMI2000.diff` after saving this block as `SMI2000.diff`, or by hand.

```diff
diff --git a/apps/host/server/index.mjs b/apps/host/server/index.mjs
index e059a657..e0411211 100644
--- a/apps/host/server/index.mjs
+++ b/apps/host/server/index.mjs
@@ -1043,6 +1043,9 @@ async function api(req, res, path, method, url) {
   if(nativeOperation&&method==='POST')return json(res,200,await native.cancel(nativeOperation[1]));
   const nativeInstance=/^\/api\/native\/instances\/([a-z0-9-]+)\/(unload|default)$/.exec(path);
   if(nativeInstance&&method==='POST')return json(res,200,nativeInstance[2]==='unload'?await native.unload(nativeInstance[1]):native.setDefault(nativeInstance[1]));
+  /* SMI2000: the Monitor's "Clear" for a model's recent requests and totals. */
+  const nativeMetricsReset=/^\/api\/native\/instances\/([a-z0-9-]+)\/metrics\/reset$/.exec(path);
+  if(nativeMetricsReset&&method==='POST'){try{return json(res,200,await native.resetMetrics(nativeMetricsReset[1]));}catch(err){return error(res,409,err.message);}}
   const nativeSettings=/^\/api\/native\/models\/([a-z0-9_.-]+)\/settings$/.exec(path);
   /* Expert mode shows what will actually run: the saved tuning's values come with the settings (null when none applies or unreadable). */
   if(nativeSettings&&method==='GET'){const settings=native.settings(nativeSettings[1]);let tuned=null;try{const t=await native.tuneEffective({model:nativeSettings[1],resourceProfile:url.searchParams.get('resourceProfile')||undefined,vision:url.searchParams.get('vision')||undefined});tuned=t.applied?t:null;}catch{tuned=null;}return json(res,200,{...settings,tuned});}
diff --git a/apps/host/server/monitor-dashboard.mjs b/apps/host/server/monitor-dashboard.mjs
index eba1f102..3fd1ea56 100644
--- a/apps/host/server/monitor-dashboard.mjs
+++ b/apps/host/server/monitor-dashboard.mjs
@@ -32,7 +32,7 @@ export function originalDashboard(raw, hardware, fallback) {
     prompt:{done:live.prompt_read??0,total:live.prompt_total??0,pct:maxPct(live.prompt_read??0,live.prompt_total)},outputTps:live.tok_s??0,promptTps:live.prefill_tok_s_mean??0,
     generation:{done:live.generated??null,total:live.max_tokens??null},
     queue:{running:live.running??(['reading','generating'].includes(live.state)?1:0),waiting:live.queued??0},context:{usedTokens:used,maxTokens:maximum,pct:maxPct(used,maximum)},
-    expertCache:{used:engine.expert_slots??0,total:engine.experts??0,memoryGb:finite(engine.expert_cache_mib)?engine.expert_cache_mib/1024:null,source:finite(engine.expert_slots)?'engine':'none'},
+    expertCache:{used:engine.expert_slots??0,total:engine.experts??(finite(engine.expert_slots)&&finite(engine.cpu_experts)?engine.expert_slots+engine.cpu_experts:0),memoryGb:finite(engine.expert_cache_mib)?engine.expert_cache_mib/1024:null,source:finite(engine.expert_slots)?'engine':'none',/* SMI2000 */totalGb:finite(engine.expert_total_mib)?engine.expert_total_mib/1024:null,ramGb:finite(engine.arena_mib)?engine.arena_mib/1024:null,ramUsed:finite(engine.ram_experts)?engine.ram_experts:null},engineInfo:engine,sampling:raw.sampling??null,
     hardware:{...fallback.hardware,gpus:readings.length?readings:fallback.hardware.gpus,gpuLoad:hw.gpu_util??fallback.hardware?.gpuLoad??null,vramUsedGb:gb(hw.gpu_mem_used)??fallback.hardware?.vramUsedGb??null,vramTotalGb:gb(hw.gpu_mem_total)??fallback.hardware.vramTotalGb,gpuTempC:hw.gpu_temp??fallback.hardware?.gpuTempC??null,powerW:hw.gpu_power??fallback.hardware?.powerW??null,powerLimitW:hw.gpu_power_limit??fallback.hardware?.powerLimitW??null,pcieRxMbs:hw.gpu_pcie_rx_mb??fallback.hardware?.pcieRxMbs??null,cpuLoad:hw.cpu??fallback.hardware.cpuLoad,ramUsedGb:gb(hw.ram_used)??fallback.hardware.ramUsedGb,ramTotalGb:gb(hw.ram_total)??fallback.hardware.ramTotalGb,diskReadGbs:finite(hw.disk_read_mb)?hw.disk_read_mb/1024:null,diskWriteMbs:hw.disk_write_mb??null},
     history:samples.length?samples:fallback.history,requests,
     totals:totals?{...fallback.totals,requests:totals.requests??0,tokensIn:totals.prompt_tokens??0,tokensOut:totals.output_tokens??0,cachedTokens:totals.reused??0,uncachedTokens:(totals.prompt_tokens??0)-(totals.reused??0),promptMs:totals.prompt_ms,decodeMs:totals.decode_ms,since:totals.since?new Date(totals.since*1000).toISOString():fallback.totals.since}:fallback.totals,
diff --git a/apps/host/server/native-control.mjs b/apps/host/server/native-control.mjs
index 01bb3637..329599af 100644
--- a/apps/host/server/native-control.mjs
+++ b/apps/host/server/native-control.mjs
@@ -591,7 +591,9 @@ return {summary:result.summary??'Deleted',removed:result.plan?.gb??null};}
  // Remove nonserializable process references from public jobs.
  const publicState=()=>{const value=state();value.jobs=value.jobs.map(({process,engineAbort,rawResult,...job})=>job);return value;};
  async function localPreview(path){if(!packaged)throw new Error('Checking local model files is part of the Strata app.');if(typeof path!=='string'||!path.trim()||[...path].some(c=>c.charCodeAt(0)<32))throw new Error('Choose a folder.');return control.importPlan(path.trim());}
- return {state:publicState,tuneStatus,tuneEffective,tuneOff,tunePresets:{list:()=>autotuner.list(),action:(kind,id,extra)=>autotuner.action(kind,id,extra),undoFor:async args=>autotuner.undoFor(tuneConfig(args).path)},refresh,localPreview,configure,setDownloadDirectory,planModelsFolder,startModelsFolder,interruptedMove,finishInterruptedMove,leaveInterruptedMove,plan,settings,save,settingsFile,openSettingsFile,operation,cancel,unload,resolveInstance,poll,logs:()=>logs.slice(),benchmarkReport:()=>readJson(join(directory,'native-benchmark.json'),null),setDefault(id){defaultId=resolveInstance(id).id;emit();return publicState();},hasLive:()=>[...instances.values()].some(i=>i.status==='ready'),readyInstances:()=>[...instances.values()].filter(i=>i.status==='ready'),defaultInstanceId:()=>defaultId,close};
+ /* SMI2000: the Monitor's "Clear": the server starts its request list and totals again; the next poll's figures replace the kept ones. */
+ async function resetMetrics(id){const instance=resolveInstance(id);if(instance.status!=='ready')throw new Error('This model is not ready.');const headers={authorization:`Bearer ${instance.key}`,'content-type':'application/json'};const r=await fetcher(`http://127.0.0.1:${instance.port}/metrics/reset`,{method:'POST',headers,body:'{}'});if(!r.ok)throw new Error(r.status===404?'This Strata server cannot clear its requests (update the server).':'The requests could not be cleared.');const fresh=await fetcher(`http://127.0.0.1:${instance.port}/metrics?requests=all`,{headers});if(fresh.ok)instance.metrics={...await fresh.json(),mcp:instance.metrics?.mcp};emit();return publicState();}
+ return {resetMetrics,state:publicState,tuneStatus,tuneEffective,tuneOff,tunePresets:{list:()=>autotuner.list(),action:(kind,id,extra)=>autotuner.action(kind,id,extra),undoFor:async args=>autotuner.undoFor(tuneConfig(args).path)},refresh,localPreview,configure,setDownloadDirectory,planModelsFolder,startModelsFolder,interruptedMove,finishInterruptedMove,leaveInterruptedMove,plan,settings,save,settingsFile,openSettingsFile,operation,cancel,unload,resolveInstance,poll,logs:()=>logs.slice(),benchmarkReport:()=>readJson(join(directory,'native-benchmark.json'),null),setDefault(id){defaultId=resolveInstance(id).id;emit();return publicState();},hasLive:()=>[...instances.values()].some(i=>i.status==='ready'),readyInstances:()=>[...instances.values()].filter(i=>i.status==='ready'),defaultInstanceId:()=>defaultId,close};
 }
 /* A DeepSeek config on AMD written by setup 0.2.8 or earlier names ONE card and marks it "asked", though setup picked it itself (strata_control/gpu_pick.py): it must not stop the app's recommended cards from applying. A card setup named for itself is the first of the usable cards, largest memory first; another card number was the person's. Roles and a "gpus_by" marker are always honoured. */
 export function setupPickedOneCard(config,hardware){
diff --git a/apps/host/src/components/SnapshotDialog.tsx b/apps/host/src/components/SnapshotDialog.tsx
index 7568ed73..780e6729 100644
--- a/apps/host/src/components/SnapshotDialog.tsx
+++ b/apps/host/src/components/SnapshotDialog.tsx
@@ -23,6 +23,18 @@ export function downloadSnapshot(data: string) {
     link.remove();
     setTimeout(() => URL.revokeObjectURL(url), 1000);
 }
+/** SMI2000: puts the picture on the clipboard (a PNG), and says so. */
+export async function copySnapshotWithNotice(data: string) {
+    try {
+        const encoded = data.slice(data.indexOf(',') + 1);
+        const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
+        await navigator.clipboard.write([new ClipboardItem({ 'image/png': new Blob([bytes], { type: 'image/png' }) })]);
+        pushToast('success', tr('Image copied'), tr('Paste it where you want with Ctrl+V.'));
+    }
+    catch {
+        pushToast('error', tr('The image was not copied'), tr('Use Download instead.'));
+    }
+}
 export function SnapshotDialog({ open, data, pending, error, onClose, onRetry }: {
     open: boolean;
     data: string;
@@ -33,7 +45,7 @@ export function SnapshotDialog({ open, data, pending, error, onClose, onRetry }:
 }) {
     const caption = 'My local AI performance with Strata';
     const project = 'https://github.com/Niko1221/Strata';
-    return <Dialog open={open} title={tr("Monitor screenshot")} onClose={onClose} actions={<><Btn onClick={onClose}>{tr("Close")}</Btn><Btn icon="download" variant="primary" disabled={!data || pending} onClick={() => saveSnapshotWithNotice(data)}>{tr("Download")}</Btn></>}>
+    return <Dialog open={open} title={tr("Monitor screenshot")} onClose={onClose} actions={<><Btn onClick={onClose}>{tr("Close")}</Btn><Btn icon="copy" disabled={!data || pending} onClick={() => void copySnapshotWithNotice(data)}>{tr("Copy to clipboard")}</Btn><Btn icon="download" variant="primary" disabled={!data || pending} onClick={() => saveSnapshotWithNotice(data)}>{tr("Download")}</Btn></>}>
     <div className="snapshot-preview" aria-busy={pending}>
       {tr(pending ? <p className="small muted" role="status">{tr("Preparing your screenshot\u2026")}</p> : data ? <img src={data} alt={tr("Captured Strata Monitor readings")}/> : <div><p className="inline-error" role="alert">{tr(error || 'Screenshot could not be prepared.')}</p><Btn onClick={onRetry}>{tr("Try again")}</Btn></div>)}
     </div>
diff --git a/apps/host/src/lib/i18n.ts b/apps/host/src/lib/i18n.ts
index 8a429c46..5abf3d21 100644
--- a/apps/host/src/lib/i18n.ts
+++ b/apps/host/src/lib/i18n.ts
@@ -44,14 +44,17 @@ import base_ru from '../locales/ru.json';
 import base_pt from '../locales/pt.json';
 import base_ko from '../locales/ko.json';
 import base_fr from '../locales/fr.json';
+/* SMI2000: the Monitor's "Memory and experts" card (components/SmiMemoryPanel.tsx). */
+import smi_en from '../locales/smi-en.json';
+import smi_es from '../locales/smi-es.json';
 
 export const LANGUAGES=[{value:'en',label:'English'},{value:'zh',label:'中文（简体）'},{value:'ja',label:'日本語'},{value:'de',label:'Deutsch'},{value:'fr',label:'Français'},{value:'es',label:'Español'},{value:'ru',label:'Русский'},{value:'pt',label:'Português'},{value:'ko',label:'한국어'}] as const;
 export type Language=typeof LANGUAGES[number]['value'];
 /* The Tune model dialog's strings live in tune-<lang>.json (apps/host/scripts/make-tune-locales.py) so other interface work does not collide with them. */
 /* The tool-server, picture and agent-result strings of the tools and images fixes live in tools-<lang>.json for the same reason. */
 /* The strings that had no key in the files above (#3479, #3825) and the 0.2.4.3 fixes live in fix-<lang>.json (scripts/make-fix-locales.py). */
-const english:Record<string,string>={...baseEnglish,...tune_en,...tools_en,...fix_en,...gh_en};
-const [zh,ja,de,es,ru,pt,ko,fr]=[{...base_zh,...tune_zh,...tools_zh,...fix_zh,...gh_zh},{...base_ja,...tune_ja,...tools_ja,...fix_ja,...gh_ja},{...base_de,...tune_de,...tools_de,...fix_de,...gh_de},{...base_es,...tune_es,...tools_es,...fix_es,...gh_es},{...base_ru,...tune_ru,...tools_ru,...fix_ru,...gh_ru},{...base_pt,...tune_pt,...tools_pt,...fix_pt,...gh_pt},{...base_ko,...tune_ko,...tools_ko,...fix_ko,...gh_ko},{...base_fr,...tune_fr,...tools_fr,...fix_fr,...gh_fr}] as Record<string,string>[];
+const english:Record<string,string>={...baseEnglish,...tune_en,...tools_en,...fix_en,...gh_en,...smi_en};
+const [zh,ja,de,es,ru,pt,ko,fr]=[{...base_zh,...tune_zh,...tools_zh,...fix_zh,...gh_zh},{...base_ja,...tune_ja,...tools_ja,...fix_ja,...gh_ja},{...base_de,...tune_de,...tools_de,...fix_de,...gh_de},{...base_es,...tune_es,...tools_es,...fix_es,...gh_es,...smi_es},{...base_ru,...tune_ru,...tools_ru,...fix_ru,...gh_ru},{...base_pt,...tune_pt,...tools_pt,...fix_pt,...gh_pt},{...base_ko,...tune_ko,...tools_ko,...fix_ko,...gh_ko},{...base_fr,...tune_fr,...tools_fr,...fix_fr,...gh_fr}] as Record<string,string>[];
 const catalogs:Record<Language,Record<string,string>>={en:english,zh,ja,de,fr,es,ru,pt,ko};
 const entities:Record<string,string>={amp:'&',lt:'<',gt:'>',quot:'"',apos:"'",nbsp:'\u00a0'};
 const decode=(text:string)=>text.replace(/&(#x[\da-f]+|#\d+|amp|lt|gt|quot|apos|nbsp);/gi,(whole,key:string)=>key[0]==='#'?String.fromCodePoint(key[1].toLowerCase()==='x'?parseInt(key.slice(2),16):parseInt(key.slice(1),10)):entities[key]??whole);
diff --git a/apps/host/src/lib/native-monitor.ts b/apps/host/src/lib/native-monitor.ts
index 16846852..e0fee865 100644
--- a/apps/host/src/lib/native-monitor.ts
+++ b/apps/host/src/lib/native-monitor.ts
@@ -40,7 +40,7 @@ export function nativeMonitorData(native:NativeState,inventory:Hardware|null,sel
   telemetry:{throughput:raw?'engine':'unavailable',thermals:hardware?'sensor':'unavailable',memory:hardware?'sensor':'unavailable',context:selected&&hasReading(used)?'reported':'unavailable'},
   prompt:{done:reading(live?.prompt_read),total:reading(live?.prompt_total),pct:hasReading(live?.prompt_read)&&live.prompt_total>0?live.prompt_read/live.prompt_total*100:0},generation:{done:value(live?.generated),total:value(live?.max_tokens)},outputTps:reading(live?.tok_s),promptTps:reading(live?.prefill_tok_s_mean),
   queue:{running:total(scoped.map(instance=>instance.metrics?.live?value(instance.metrics.live.running)??(['reading','generating'].includes(instance.metrics.live.state)?1:0):null)),waiting:total(scoped.map(instance=>value(instance.metrics?.live?.queued)))},
-  context:{usedTokens:used,maxTokens:maxContext,pct:hasReading(used)&&maxContext>0?used/maxContext*100:NaN},expertCache:{used:reading(engine?.expert_slots),total:reading(engine?.experts),memoryGb:hasReading(engine?.expert_cache_mib)?engine.expert_cache_mib/1024:null,source:engine&&hasReading(engine.expert_slots)?'engine':'unavailable'},
+  context:{usedTokens:used,maxTokens:maxContext,pct:hasReading(used)&&maxContext>0?used/maxContext*100:NaN},expertCache:{used:reading(engine?.expert_slots),total:reading(engine?.experts??(hasReading(engine?.expert_slots)&&hasReading(engine?.cpu_experts)?engine.expert_slots+engine.cpu_experts:null)),memoryGb:hasReading(engine?.expert_cache_mib)?engine.expert_cache_mib/1024:null,source:engine&&hasReading(engine.expert_slots)?'engine':'unavailable',/* SMI2000 */totalGb:hasReading(engine?.expert_total_mib)?engine.expert_total_mib/1024:null,ramGb:hasReading(engine?.arena_mib)?engine.arena_mib/1024:null,ramUsed:value(engine?.ram_experts)},engineInfo:engine??null,sampling:selected?.metrics?.sampling??null,
   hardware:{gpus,cpuLoad:value(hardware?.cpu_pct??hardware?.cpu),gpuLoad:value(hardware?.gpu_util),vramUsedGb:bytes(hardware?.gpu_mem_used),vramTotalGb:gpuCapacity(inventory?.gpus??[]),ramUsedGb:bytes(hardware?.ram_used),ramTotalGb:bytes(hardware?.ram_total)??inventory?.memory.totalGb??NaN,gpuTempC:value(hardware?.gpu_temp),tempLimitC:90,powerW:value(hardware?.gpu_power),powerLimitW:value(hardware?.gpu_power_limit),pcieRxMbs:value(hardware?.gpu_pcie_rx_mb),diskReadGbs:hasReading(hardware?.disk_read_mb)?hardware.disk_read_mb/1024:null,diskWriteMbs:value(hardware?.disk_write_mb)},history,requests,
   mcp:raw?.mcp?{servers:raw.mcp.servers??[],tools:raw.mcp.tools??0}:undefined,
   totals:{tokensIn:tokenTotal('prompt_tokens'),tokensOut:tokenTotal('output_tokens'),requests:tokenTotal('requests'),avgTps:NaN,since:selected?.loadedAt??instances[0]?.loadedAt??new Date().toISOString(),ttftMs:NaN,cachedTokens:tokenTotal('reused'),uncachedTokens:tokenTotal('prompt_tokens')-tokenTotal('reused'),promptMs:hasReading(tokenTotal('prompt_ms'))?tokenTotal('prompt_ms'):undefined,decodeMs:hasReading(tokenTotal('decode_ms'))?tokenTotal('decode_ms'):undefined}
diff --git a/apps/host/src/lib/types.ts b/apps/host/src/lib/types.ts
index 1e477540..635ae523 100644
--- a/apps/host/src/lib/types.ts
+++ b/apps/host/src/lib/types.ts
@@ -277,7 +277,10 @@ export interface MonitorState {
     pct: number;
     parts?: { system: number; tools: number; messages: number };
   };
-  expertCache: { used: number; total: number; memoryGb?:number|null; source?:string };
+  expertCache: { used: number; total: number; memoryGb?:number|null; source?:string; /* SMI2000 */ totalGb?:number|null; ramGb?:number|null; ramUsed?:number|null };
+  /** SMI2000: the selected model's engine facts (/metrics `engine`) and its default sampling (/metrics `sampling`). */
+  engineInfo?: Record<string, any> | null;
+  sampling?: { config?: Record<string, any>; thinking?: Record<string, any>; instruct?: Record<string, any> } | null;
   hardware: {
     gpus?: GpuReading[];
     cpus?: CpuReading[];
diff --git a/apps/host/src/main.tsx b/apps/host/src/main.tsx
index 446e060a..a44d1030 100644
--- a/apps/host/src/main.tsx
+++ b/apps/host/src/main.tsx
@@ -19,6 +19,7 @@ createRoot(el).render(
 
 import './styles/annotation-fixes.css';
 import './styles/monitor.css';
+import './styles/smi-monitor.css';
 import './styles/time-skip.css';
 
 import './styles/model-run.css';
diff --git a/apps/host/src/screens/Monitor.tsx b/apps/host/src/screens/Monitor.tsx
index d75db79a..1da41b69 100644
--- a/apps/host/src/screens/Monitor.tsx
+++ b/apps/host/src/screens/Monitor.tsx
@@ -20,6 +20,8 @@ import { useKept } from '../lib/kept-state';
 import type { Hardware, MonitorState, ServeRequest } from '../lib/types';
 import { useNative, nativeRouting, type NativeState } from '../lib/native';
 import { nativeMonitorData, withHostSensors } from '../lib/native-monitor';
+import { SmiMemoryPanel } from '../components/SmiMemoryPanel';
+import { SmiClearRequests, SmiKvInfo, SmiSamplingCard, SmiVramBar } from '../components/SmiMonitorExtras';
 const number = (value: number | null | undefined, digits = 0) => hasReading(value) ? value.toLocaleString(localeTag(), { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
 const percent = (used: number | null | undefined, total: number | null | undefined) => hasReading(used) && hasReading(total) && total > 0 ? Math.max(0, Math.min(100, used / total * 100)) : null;
 const sourceLabel = (source: string | undefined) => source === 'demo' ? 'Illustrative' : source === 'sensor' || source === 'engine' || source === 'reported' ? 'Measured' : source === 'estimated' ? 'Estimated' : 'Unavailable';
@@ -189,7 +191,7 @@ function LegacyMonitorScreen({ native }: {
     <div className="monitor-metrics">
       {tr(!multipleScope ? <MonitorMetric label={tr("Speed")} icon="gauge" value={number(decode, 1)} unit={hasReading(decode) ? 't/s' : ''} source={throughput} sub={writing ? 'Decode now' : last ? 'Decode last request' : 'Decode'} secondary={<div><div className="st-metric__value">{tr(number(prefill))}{tr(hasReading(prefill) ? <span className="st-metric__unit">{tr("t/s")}</span> : null)}</div><div className="monitor-metric__sub xs muted">{tr(data.phase === 'reading' ? 'Prefill now' : writing ? 'Prefill this request' : last ? 'Prefill last request' : 'Prefill')}</div></div>} chart={<ReferenceSpark label={tr("Speed")} data={history.map(s => throughputKnown ? s.outputTps : null)} secondary={{ data: history.map(s => throughputKnown ? s.promptTps : null), label: 'Prefill', tone: 'var(--st-chart-prefill)' }} tone="var(--st-chart-decode)" unit="t/s" timestamps={timestamps}/>}/> : null)}
       <MonitorMetric label={tr("GPU load")} icon="gpu" value={number(gpuLoad)} unit={hasReading(gpuLoad) ? '%' : ''} source={gpuSource('utilizationPct', 'mean')} sub={whose('utilizationPct', 'mean') ?? gpuName} chart={chart('GPU load', historyGpu('utilizationPct', 'gpuLoad', 'mean'), '%', 100)}/>
-      <MonitorMetric label={tr(shared ? 'Shared GPU memory' : 'VRAM')} icon="layers" value={number(vram, 1)} unit={hasReading(vram) ? vramTotal ? `/ ${number(vramTotal, 0)} GB` : 'GB' : ''} source={gpuSource('memoryUsedGb')} sub={whose('memoryUsedGb') ?? (shared ? 'Same memory pool as system RAM' : multipleGpus ? 'Total across selected GPUs' : cacheKnown && devices.length === 1 ? `${number(data.expertCache.used)} experts cached` : 'GPU memory')} chart={chart(shared ? 'Shared GPU memory' : 'VRAM', historyGpu('memoryUsedGb', 'vramUsedGb'), 'GB', vramTotal)}/>
+      <MonitorMetric label={tr(shared ? 'Shared GPU memory' : 'VRAM')} icon="layers" value={number(vram, 1)} unit={hasReading(vram) ? vramTotal ? `/ ${number(vramTotal, 0)} GB` : 'GB' : ''} source={gpuSource('memoryUsedGb')} sub={whose('memoryUsedGb') ?? (shared ? 'Same memory pool as system RAM' : multipleGpus ? 'Total across selected GPUs' : cacheKnown && devices.length === 1 ? `${number(data.expertCache.used)} experts cached` : 'GPU memory')} chart={<>{chart(shared ? 'Shared GPU memory' : 'VRAM', historyGpu('memoryUsedGb', 'vramUsedGb'), 'GB', vramTotal)}{selectedGpus.length === 1 && (selectedInstance?.gpuIds.length ?? 0) <= 1 ? <SmiVramBar engine={data.engineInfo} usedGb={vram} totalGb={vramTotal}/> : null}</>}/>
       <MonitorMetric label={tr("GPU temp")} icon="thermometer" value={number(temperature)} unit={hasReading(temperature) ? '°C' : ''} source={gpuSource('temperatureC', 'max')} sub={whose('temperatureC', 'max') ?? (multipleGpus ? 'Hottest selected GPU' : selectedGpus[0]?.name ?? (selectedInstance && !selectedInstance.gpuIds.length ? native ? 'Automatic GPU placement' : 'CPU only · no GPU assigned' : 'No GPU detected'))} chart={chart('GPU temperature', historyGpu('temperatureC', 'gpuTempC', 'max'), '°C', 90, 'var(--st-warn)')}/>
       <MonitorMetric label={tr("Power")} icon="bolt" value={number(power)} unit={hasReading(power) ? 'W' : ''} source={gpuSource('powerW')} sub={whose('powerW') ?? (hasReading(powerLimit) ? `of ${number(powerLimit)} W limit` : multipleGpus ? 'Total across selected GPUs' : 'GPU power draw')} chart={chart('GPU power', historyGpu('powerW', 'powerW'), 'W', powerLimit ?? undefined)}/>
       <MonitorMetric label={tr("PCIe")} icon="link" value={hasReading(gen) ? `Gen${gen}` : multipleGpus ? number(pcieRx, 1) : '—'} unit={hasReading(gen) && hasReading(width) ? `x${width}` : multipleGpus && hasReading(pcieRx) ? 'MB/s' : ''} source={hasReading(gen) ? sourceLabel(singleReading?.source) : gpuSource('pcieRxMbs')} sub={hasReading(pcieRx) ? `To GPU ${number(pcieRx, 1)} MB/s${whose('pcieRxMbs') ? ` · ${whose('pcieRxMbs')}` : ''}${hasReading(gen) && hasReading(singleReading?.pcieGen) && singleReading.pcieGen < gen ? ` · idle Gen${singleReading.pcieGen}` : ''}` : shared ? 'Shared memory · no PCIe link' : multipleGpus ? 'Total transfer to selected GPUs' : 'Transfer to GPU unavailable'} chart={chart('PCIe transfer', historyGpu('pcieRxMbs', 'pcieRxMbs'), 'MB/s', undefined, 'var(--st-info)')}/>
@@ -197,14 +199,16 @@ function LegacyMonitorScreen({ native }: {
       <MonitorMetric label={tr("Disk read")} icon="disk" value={number(hasReading(data.hardware.diskReadGbs) ? data.hardware.diskReadGbs >= 1 ? data.hardware.diskReadGbs : data.hardware.diskReadGbs * 1024 : null, 1)} unit={hasReading(data.hardware.diskReadGbs) ? data.hardware.diskReadGbs >= 1 ? 'GB/s' : 'MB/s' : ''} source={hasReading(data.hardware.diskReadGbs) ? 'Measured' : 'Unavailable'} sub={hasReading(data.hardware.diskWriteMbs) ? `Write ${number(data.hardware.diskWriteMbs, 1)} MB/s` : 'Read throughput'} chart={chart('Disk read', history.map(s => hasReading(s.diskReadGbs) ? s.diskReadGbs * 1024 : null), 'MB/s', undefined, 'var(--st-info)')}/>
     </div>
     {tr(readingsStrip.length ? <section className="st-card monitor-readings" aria-label={tr("Energy and temperature readings")}><dl>{readingsStrip.map(item => <div key={item.label}><dt>{tr(item.label)}</dt><dd className="num">{item.value}</dd></div>)}</dl></section> : null)}
+    <div className="smi-row" data-single={!data.engineInfo}>{data.engineInfo ? <SmiSamplingCard sampling={data.sampling}/> : null}<SmiMemoryPanel data={data}/></div>
     <div className="monitor-context-requests">
       <section className="st-card monitor-context"><h2 className="h4">{tr("Context fill")}</h2><div className="monitor-context__gauge"><Gauge value={contextPct ?? 0} label={tr(hasReading(contextPct) ? `${Math.round(contextPct)}%` : '—')} sub={contextKnown ? `${tokens(contextUsed)} / ${contextLabel(data.context.maxTokens)}` : selection === 'all' && (runtime?.instances.length ?? 0) > 1 ? 'Select a model to see context' : 'No recorded request usage'}/></div>
+        <SmiKvInfo engine={data.engineInfo}/>
         <div className="monitor-bar-reading"><span>{tr("Experts in VRAM")}</span><span className="muted num">{tr(cacheKnown ? `${number(data.expertCache.used)}${hasReading(cacheMemory) ? ` · ${number(cacheMemory, 1)} GB` : ''}` : '—')}</span></div><ReadingBar label={tr("Experts in VRAM")} value={cachePercent}/>
         <div className="monitor-bar-reading"><span>{tr("System RAM")}</span><span className="muted num">{tr(hasReading(data.hardware.ramUsedGb) ? `${number(data.hardware.ramUsedGb, 1)} / ${number(data.hardware.ramTotalGb)} GB` : '—')}</span></div><ReadingBar label={tr("System RAM")} value={ramPct} tone={hasReading(ramPct) && ramPct > 92 ? 'danger' : 'accent'}/>
         <div className="monitor-bar-reading"><span>{tr("GPU temperature")}</span><span className="muted num">{tr(number(temperature))}{tr(hasReading(temperature) ? ' °C' : '')}</span></div><ReadingBar label={tr("GPU temperature")} value={temperature} tone="warn"/>
       </section>
-      <section className="st-card monitor-requests"><div className="section-toolbar"><h2 className="h4">{tr("Recent requests")}</h2>{tr(requests.length > 12 ? <Btn size="sm" onClick={() => setShowAll(!showAll)}>{tr(showAll ? 'Show fewer' : `Show all (${requests.length})`)}</Btn> : null)}</div>
-        <div className="st-tablewrap monitor-request-table" data-expanded={showAll}><table className="st-table"><thead><tr><th>{tr("Time")}</th><th>{tr("Status")}</th>{tr(multipleModels ? <th>{tr("Model")}</th> : null)}<th className="num">{tr("Prompt")}</th><th className="num">{tr("Reused")}</th><th className="num">{tr("Output")}</th><th className="num">{tr("Tok/s")}</th><th className="num" title={tr("Experts already in GPU memory; PCIe share is shown separately when reported.")}>{tr("VRAM hit rate")}</th><th className="num">{tr("Duration")}</th></tr></thead><tbody>{tr(visibleRequests.length ? visibleRequests.map(r => <tr key={r.id}><td>{tr(new Date(r.at).toLocaleTimeString(localeTag(), { hour: '2-digit', minute: '2-digit', second: '2-digit' }))}</td><td><span className="request-status" data-state={r.state}>{tr(requestStatus(r))}</span>{tr(r.usageSource === 'demo' ? <span className="monitor-request-source xs muted">{tr("Illustrative")}</span> : null)}</td>{tr(multipleModels ? <td><span className="xs">{tr(r.model)}</span></td> : null)}<td className="num">{tr(number(r.tokensIn))}</td><td className="num">{tr(number(r.cachedTokens))}</td><td className="num">{tr(number(r.tokensOut))}</td><td className="num">{tr(number(r.tps, 1))}</td><td className="num">{tr(hasReading(r.hitRate) ? `${number(r.hitRate * 100, 1)}%` : '—')}{tr(hasReading(r.pcieShare) && r.pcieShare > 0 ? <span className="xs muted"> +{tr(number(r.pcieShare * 100, 1))}{tr("% PCIe")}</span> : null)}</td><td className="num">{tr(r.durationMs ? duration(r.durationMs) : '—')}</td></tr>) : <tr><td colSpan={multipleModels ? 9 : 8} className="muted">{tr("No requests yet")}</td></tr>)}</tbody></table></div>
+      <section className="st-card monitor-requests"><div className="section-toolbar"><h2 className="h4">{tr("Recent requests")}</h2><span className="smi-toolbar-actions"><SmiClearRequests ids={native ? (selectedInstance ? [selectedInstance.id] : runtime?.instances.map(i => i.id) ?? []) : []}/>{tr(requests.length > 12 ? <Btn size="sm" onClick={() => setShowAll(!showAll)}>{tr(showAll ? 'Show fewer' : `Show all (${requests.length})`)}</Btn> : null)}</span></div>
+        <div className="st-tablewrap monitor-request-table" data-expanded={showAll}><table className="st-table"><thead><tr><th>{tr("Time")}</th><th>{tr("Status")}</th>{tr(multipleModels ? <th>{tr("Model")}</th> : null)}<th className="num">{tr("Prompt")}</th><th className="num">{tr("Reused")}</th><th className="num">{tr("Output")}</th><th className="num">{tr("Tok/s")}</th><th className="num">{tr("Prefill tok/s")}</th><th className="num" title={tr("Experts already in GPU memory; PCIe share is shown separately when reported.")}>{tr("VRAM hit rate")}</th><th className="num">{tr("Duration")}</th></tr></thead><tbody>{tr(visibleRequests.length ? visibleRequests.map(r => <tr key={r.id}><td>{tr(new Date(r.at).toLocaleTimeString(localeTag(), { hour: '2-digit', minute: '2-digit', second: '2-digit' }))}</td><td><span className="request-status" data-state={r.state}>{tr(requestStatus(r))}</span>{tr(r.usageSource === 'demo' ? <span className="monitor-request-source xs muted">{tr("Illustrative")}</span> : null)}</td>{tr(multipleModels ? <td><span className="xs">{tr(r.model)}</span></td> : null)}<td className="num">{tr(number(r.tokensIn))}</td><td className="num">{tr(number(r.cachedTokens))}</td><td className="num">{tr(number(r.tokensOut))}</td><td className="num">{tr(number(r.tps, 1))}</td><td className="num">{tr(number(r.promptTps, 0))}</td><td className="num">{tr(hasReading(r.hitRate) ? `${number(r.hitRate * 100, 1)}%` : '—')}{tr(hasReading(r.pcieShare) && r.pcieShare > 0 ? <span className="xs muted"> +{tr(number(r.pcieShare * 100, 1))}{tr("% PCIe")}</span> : null)}</td><td className="num">{tr(r.durationMs ? duration(r.durationMs) : '—')}</td></tr>) : <tr><td colSpan={multipleModels ? 10 : 9} className="muted">{tr("No requests yet")}</td></tr>)}</tbody></table></div>
         {tr(data.totals.requests > 0 ? <p className="small muted monitor-request-totals">{requestSummary({ requests: data.totals.requests, output: outputSpeed, reading: promptSpeed, reused: data.totals.cachedTokens }, { number, tokens })}</p> : null)}
       </section>
     </div>
diff --git a/apps/host/tests/locale-gaps.test.mjs b/apps/host/tests/locale-gaps.test.mjs
index 1dac2c78..fa4f87d0 100644
--- a/apps/host/tests/locale-gaps.test.mjs
+++ b/apps/host/tests/locale-gaps.test.mjs
@@ -12,7 +12,7 @@ const { tr, setTranslationLanguage } = await import(`data:text/javascript;base64
 
 test('every tr("...") literal and every command hint has a key in the English catalog, so no language is stuck in English (#3479)', async () => {
   const known = new Set();
-  for (const name of ['en', 'tune-en', 'tools-en', 'fix-en', 'gh-en']) for (const v of Object.values(await read(name))) known.add(String(v));
+  for (const name of ['en', 'tune-en', 'tools-en', 'fix-en', 'gh-en', 'smi-en']) for (const v of Object.values(await read(name))) known.add(String(v));
   const missing = [];
   const scan = async dir => {
     for (const entry of await readdir(dir, { withFileTypes: true })) {
diff --git a/serve/server.py b/serve/server.py
index 768f6bcc..69785e44 100644
--- a/serve/server.py
+++ b/serve/server.py
@@ -911,6 +911,54 @@ def parse_engine_stage_line(line: str) -> list[dict] | None:
     return stages
 
 
+# SMI2000: the pack's expert totals (count and size) so the Monitor can split the experts into VRAM, RAM and disk.
+def pack_expert_totals(args: list[str]) -> dict:
+    try:
+        pack = Path(args[args.index("--pack") + 1])
+        with open(pack / "native_experts.txt", encoding="utf-8") as f:
+            header, layers = f.readline(), sum(1 for line in f if line.strip() and not line.startswith("#"))
+        n_expert, total = re.search(r"n_expert (\d+)", header), re.search(r"total (\d+)", header)
+        if not (n_expert and total and layers):
+            return {}
+        return {"experts": layers * int(n_expert.group(1)), "expert_total_mib": int(total.group(1)) >> 20}
+    except (ValueError, IndexError, OSError):
+        return {}
+
+
+# SMI2000: what the engine's start lines say it put in VRAM (MiB), for the Monitor's VRAM bar.  A line it did not
+# write leaves its part out; nothing is guessed here.
+VRAM_LOG_PARTS = (
+    ("weights", r"(\d+(?:\.\d+)?) MiB of weights loaded from", 1),
+    ("weights", r"native projection matrices, (\d+(?:\.\d+)?) MiB of weights", 1),
+    ("weights", r"(\d+(?:\.\d+)?) GiB of Q8_0 hyper-connection projections", 1024),
+    ("draft", r"draft layer loaded, (\d+(?:\.\d+)?) MiB of VRAM", 1),
+    ("draft", r"draft head over \d+ tokens \((\d+(?:\.\d+)?) MiB\)", 1),
+    ("buffers", r"window up to \d+ tokens, (\d+(?:\.\d+)?) MiB of device buffers", 1),
+)
+
+
+def vram_parts_from_log(text: str) -> dict:
+    parts: dict = {}
+    for name, pattern, scale in VRAM_LOG_PARTS:
+        found = re.findall(pattern, text)
+        if found:                                     # the last one: a start that retried writes the lines again
+            parts[name] = round(parts.get(name, 0) + float(found[-1]) * scale)
+    return parts
+
+
+# SMI2000: KV bytes per context token and attention layer, and the attention layers (12 QSA + the draft layer) of
+# Qwen3.8-Flash-Next; the same figures as setup.UD_FIT_KV_BYTES and strata_control/vram_fit.py (5.4% measured over).
+SMI_KV_BYTES = {"fp16": 2112, "int8": 1056, "q4_0": 576, "k8v4": 816}
+SMI_KV_LAYERS = {"qwen4exp": 13}
+
+
+def kv_estimate_mib(arch, kv, context) -> int | None:
+    layers, cell = SMI_KV_LAYERS.get(arch or ""), SMI_KV_BYTES.get(str(kv or ""))
+    if not layers or not cell or not isinstance(context, int) or context <= 0:
+        return None
+    return round(121 + context * layers * cell * 1.054 / 2 ** 20)
+
+
 def read_engine_stage_log(path: str | None, start: int, limit: int = 2 << 20) -> list[dict]:
     """Read this engine start only; oversized/unavailable logs leave placement unknown."""
     if not path or start < 0:
@@ -1161,6 +1209,12 @@ class StrataEngine:
             self.info.update(drafting.engine_info(args, start_log_text(log, log_start), drafting.take_note()))
         except Exception:                               # noqa: BLE001 - a note must never stop a start
             pass
+        for k, v in pack_expert_totals(args).items():   # SMI2000: the engine's own figures win
+            self.info.setdefault(k, v)
+        try:                                            # SMI2000: the Monitor's VRAM bar
+            self.info["vram_parts"] = vram_parts_from_log(start_log_text(log, log_start))
+        except Exception:                               # noqa: BLE001 - a Monitor figure must never stop a start
+            pass
         self.ended = False                              # READY: alive from here (restart() set it True, #344)
         # the engine's stdout on a thread, so a request can wait with a timeout (heartbeats, cancel checks)
         self.lines: queue.Queue = queue.Queue()
@@ -4952,6 +5006,14 @@ class Service:
                     | {"wallclock_s": r.get("wallclock_s", round(time.perf_counter() - r["_clock"], 3))}
                     for r in reversed(records)]
 
+    def reset_metrics(self) -> None:
+        """SMI2000: POST /metrics/reset - the Monitor starts its request list and totals again.  Only the /metrics view
+        is reset: /v1/status and the Prometheus counters keep counting from the server's start."""
+        with self.status_lock:
+            self.history.clear()
+            self.metrics_base = {**{k: v for k, v in self.totals.items() if isinstance(v, (int, float))},
+                                 "since": time.time()}
+
     def metrics(self, all_requests=False) -> dict:
         """GET /metrics: what the Monitor tab shows - the engine's facts, what it is doing, the last requests, and
         the hardware (with a minute of history per series)."""
@@ -4961,6 +5023,10 @@ class Service:
                 s = {**s, **dict(list(self.live_reqs.values())[-1][0]), "queued": s.get("queued", 0)}
             hist = list(self.history)
             totals = dict(self.totals)
+            base = getattr(self, "metrics_base", None)    # SMI2000: the totals since the Monitor's last "Clear"
+            if base:
+                totals = {k: (base["since"] if k == "since" else v - base.get(k, 0))
+                          if isinstance(v, (int, float)) else v for k, v in totals.items()}
         now = time.time()
         progress = getattr(self.engine, "progress", None)
         if s.get("busy") and s.get("first_token") is None:
@@ -5002,13 +5068,20 @@ class Service:
             live["split_serve"] = self.engine.split_view()
         engine = {"model": self.model, "max_context": self.reported_ctx(), "images": self.vision is not None,
                   **dict(getattr(self.engine, "info", {}) or {})}
+        engine["kv_mib_est"] = kv_estimate_mib(self.model_arch, engine.get("kv"), engine.get("context"))   # SMI2000
         tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "history": {}, "static": {}}
         parked = self.conv_log.poll(getattr(self.engine, "log_path", None), getattr(self.engine, "log_start", None))
         return {"engine": engine, "live": live, "requests": hist[::-1][:None if all_requests else 12],
                 "conversation_cache": conversation_cache_view(engine, hist, totals, parked),
                 "requests_kept": len(hist), "totals": totals, "hardware": tel["now"],
                 "hardware_static":
-                tel["static"], "history": tel["history"], "time": now}
+                tel["static"], "history": tel["history"], "time": now,
+                # SMI2000: the sampling a request gets for the fields it leaves out (model card, config, Chat settings)
+                "sampling": {"config": dict(self.sampling_defaults),
+                             "thinking": {**model_card_sampling(self.model_arch, True), **self.sampling_defaults,
+                                          **self.shared},
+                             "instruct": {**model_card_sampling(self.model_arch, False), **self.sampling_defaults,
+                                          **self.shared}}}
 
     def prometheus(self) -> str:
         """GET /metrics/prometheus: the totals and what is running in Prometheus' text format, under llama-server's
@@ -7425,6 +7498,11 @@ def make_handler(svc: Service):
             if path == "/config":
                 self._config_post()
                 return
+            if path == "/metrics/reset":                      # SMI2000: the Monitor's "Clear" (requests and totals)
+                if self._own_page("the Monitor's requests can be cleared"):
+                    svc.reset_metrics()
+                    self._json(200, {"status": "cleared"})
+                return
             if path in ("/unload", "/load") and not self._control_body():
                 return
             # JSON from Strata's own page only, as /settings: else a plain form POST from any site unloads the model
```

## Appendix B: the new files

### `apps/host/src/lib/smi-expert-tiers.ts`

```ts
/** SMI2000: where the model's experts are - VRAM (the engine's GPU cache), RAM (its resident copy: Qwen arena_mib,
 *  DeepSeek ram_experts) and the NVMe disk (the rest, read on demand). */
import { hasReading } from './monitor-hardware';
import type { MonitorState } from './types';

const clamp = (value: number, max: number) => Math.max(0, Math.min(max, value));
export type ExpertTier = { count: number | null; gb: number | null; pct: number; approx: boolean };
export type ExpertTiers = { vram: ExpertTier; ram: ExpertTier; disk: ExpertTier; totalCount: number; totalGb: number | null };

export function expertTiers(cache: MonitorState['expertCache']): ExpertTiers | null {
    if (!['engine', 'demo'].includes(cache.source ?? '') || !hasReading(cache.used) || !hasReading(cache.total) || cache.total <= 0)
        return null;
    const totalCount = cache.total, totalGb = hasReading(cache.totalGb) && cache.totalGb > 0 ? cache.totalGb : null;
    const perGb = totalGb ? totalGb / totalCount : null;
    const vramCount = clamp(cache.used, totalCount);
    const vramGb = hasReading(cache.memoryGb) ? cache.memoryGb : perGb ? vramCount * perGb : null;
    let ramCount: number, ramGb: number | null, ramApprox = false;
    if (hasReading(cache.ramUsed)) {
        ramCount = clamp(cache.ramUsed, totalCount - vramCount);
        ramGb = perGb ? ramCount * perGb : null;
    }
    else if (hasReading(cache.ramGb) && totalGb && perGb && hasReading(vramGb)) {
        // a full RAM copy (no --resident-budget-gib) also holds the GPU's experts: they are counted once, in VRAM
        ramGb = clamp(cache.ramGb, totalGb - vramGb);
        ramCount = Math.round(ramGb / perGb);
        ramApprox = true;
    }
    else
        return null;
    const diskCount = Math.max(0, totalCount - vramCount - ramCount);
    const diskGb = totalGb && hasReading(vramGb) && hasReading(ramGb) ? Math.max(0, totalGb - vramGb - ramGb) : null;
    const share = (count: number, gb: number | null) => totalGb && hasReading(gb) ? gb / totalGb * 100 : count / totalCount * 100;
    return {
        vram: { count: vramCount, gb: vramGb, pct: share(vramCount, vramGb), approx: false },
        ram: { count: ramCount, gb: ramGb, pct: share(ramCount, ramGb), approx: ramApprox },
        disk: { count: diskCount, gb: diskGb, pct: share(diskCount, diskGb), approx: ramApprox },
        totalCount, totalGb,
    };
}

/** SMI2000: the KV cache's format (engine `kv`), its size (the server's `kv_mib_est`, an estimate) and whether it streams from RAM. */
const KV_FORMATS: Record<string, [string, string]> = {
    int8: ['8-bit keys and values.', 'The default: precise.'],
    q4_0: ['4-bit keys and values, rotated before rounding.', 'Half the memory of 8-bit; less precise on long documents.'],
    k8v4: ['8-bit keys, 4-bit values.', '23% less memory than 8-bit, so more experts fit in VRAM; prompts are 2-5% slower.'],
    fp16: ['16-bit keys and values.', 'Twice the memory of 8-bit.'],
    fp4: ['DeepSeek layout: the compressed cache in FP4, the recent window in FP8.', ''],
};
export type KvDetails = { mode: string; format: string; note: string; gb: number | null; kbPerToken: number | null; streaming: boolean };
export function kvDetails(engine: Record<string, any> | null | undefined): KvDetails | null {
    const mode = typeof engine?.kv === 'string' ? engine.kv : null;
    if (!mode)
        return null;
    const [format, note] = KV_FORMATS[mode] ?? [mode, ''];
    const est = hasReading(engine?.kv_mib_est) ? engine.kv_mib_est as number : null, context = hasReading(engine?.context) ? engine.context as number : null;
    return { mode, format, note, gb: est !== null ? est / 1024 : null, kbPerToken: est !== null && context ? Math.max(0, est - 121) * 1048576 / context / 1000 : null, streaming: hasReading(engine?.kv_resident) && engine.kv_resident > 0 };
}

/** SMI2000: what fills the GPU's memory, in GB: the engine's start figures, the KV estimate, the rest of what is used, and the free memory. */
export type VramSegment = { id: string; label: string; short: string; gb: number; pct: number; approx: boolean };
export function vramSegments(engine: Record<string, any> | null | undefined, usedGb: number | null | undefined, totalGb: number | null | undefined): VramSegment[] | null {
    if (!engine || !hasReading(usedGb) || !hasReading(totalGb) || totalGb <= 0)
        return null;
    const parts = (engine.vram_parts ?? {}) as Record<string, number>;
    const experts = engine.vram?.expert_cache_mib ?? engine.expert_cache_mib;
    let kv = hasReading(engine.kv_mib_est) ? engine.kv_mib_est as number : null;
    if (kv !== null && hasReading(engine.kv_resident) && engine.kv_resident > 0 && hasReading(engine.context) && engine.context > 0)
        kv = kv * Math.min(1, engine.kv_resident / engine.context);
    const known = [
        { id: 'weights', label: 'Model weights', short: 'Weights', mib: parts.weights, approx: false },
        { id: 'draft', label: 'Draft model (MTP)', short: 'MTP draft', mib: parts.draft, approx: false },
        { id: 'experts', label: 'Expert cache', short: 'Experts', mib: experts, approx: false },
        { id: 'kv', label: 'KV cache (context)', short: 'KV cache', mib: kv, approx: true },
        { id: 'buffers', label: 'Verify buffers', short: 'Buffers', mib: parts.buffers, approx: false },
    ].filter((p): p is { id: string; label: string; short: string; mib: number; approx: boolean } => hasReading(p.mib) && p.mib > 0);
    if (!known.some(p => p.id !== 'kv'))
        return null;
    const total = totalGb * 1024, used = Math.min(usedGb * 1024, total), sum = known.reduce((n, p) => n + p.mib, 0);
    const scale = sum > used && sum > 0 ? used / sum : 1;
    const out = [...known.map(p => ({ ...p, mib: p.mib * scale })), { id: 'other', label: 'Other (work memory, other programs)', short: 'Other', mib: Math.max(0, used - sum), approx: false }, { id: 'free', label: 'Free', short: 'Free', mib: Math.max(0, total - used), approx: false }];
    return out.map(p => ({ id: p.id, label: p.label, short: p.short, gb: p.mib / 1024, pct: p.mib / total * 100, approx: p.approx }));
}
```

### `apps/host/src/components/SmiMemoryPanel.tsx`

```tsx
/** SMI2000: the Monitor's "Memory and experts" card - system RAM, and where the model's experts are (VRAM, RAM, NVMe). */
import { localeTag, tr } from '../lib/i18n';
import { hasReading } from '../lib/monitor-hardware';
import { expertTiers, type ExpertTier as Tier } from '../lib/smi-expert-tiers';
import type { MonitorState } from '../lib/types';
import { ReferenceSpark } from './charts';
import { Bar } from './ui';

const number = (value: number | null | undefined, digits = 0) => hasReading(value) ? value.toLocaleString(localeTag(), { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
const clamp = (value: number, max: number) => Math.max(0, Math.min(max, value));

export function SmiMemoryPanel({ data }: { data: MonitorState }) {
    const hw = data.hardware, ramUsed = hw.ramUsedGb, ramTotal = hw.ramTotalGb;
    const ramPct = hasReading(ramUsed) && hasReading(ramTotal) && ramTotal > 0 ? clamp(ramUsed / ramTotal * 100, 100) : null;
    const tiers = expertTiers(data.expertCache);
    const rows = tiers ? [
        { id: 'vram', label: 'VRAM', tier: tiers.vram },
        { id: 'ram', label: 'RAM', tier: tiers.ram },
        { id: 'disk', label: 'NVMe disk', tier: tiers.disk },
    ] : [];
    const count = (t: Tier) => hasReading(t.count) ? `${t.approx ? '≈' : ''}${number(t.count)} ${tr('experts')}` : '—';
    return <section className="st-card smi-memory" aria-label={tr('Memory and experts')}>
      <h2 className="h4">{tr('Memory and experts')}</h2>
      <div className="smi-memory__grid">
        <div className="smi-memory__ram">
          <div className="monitor-bar-reading"><span>{tr('System RAM')}</span><span className="muted num">{hasReading(ramUsed) ? `${number(ramUsed, 1)} / ${number(ramTotal)} GB` : '—'}</span></div>
          {hasReading(ramPct) ? <Bar label={tr('System RAM')} value={ramPct} tone={ramPct > 92 ? 'danger' : 'accent'}/> : <div className="st-progress"/>}
          <div className="smi-memory__facts xs muted">
            <span>{hasReading(ramPct) ? `${number(ramPct)}% ${tr('used')}` : ''}</span>
            <span>{hasReading(ramUsed) && hasReading(ramTotal) ? `${number(Math.max(0, ramTotal - ramUsed), 1)} GB ${tr('free')}` : ''}</span>
            {tiers && hasReading(tiers.ram.gb) ? <span>{tr('Experts kept in RAM')}: {number(tiers.ram.gb, 1)} GB</span> : null}
          </div>
          <ReferenceSpark label={tr('System RAM')} data={data.history.map(s => s.ramUsedGb)} unit="GB" timestamps={data.history.map(s => s.t)} maximum={hasReading(ramTotal) ? ramTotal : undefined} tone="var(--st-info)"/>
        </div>
        <div className="smi-memory__experts">
          <div className="monitor-bar-reading"><span>{tr('Expert placement')}</span><span className="muted num">{tiers ? `${tr('Total')} ${number(tiers.totalCount)} ${tr('experts')}${hasReading(tiers.totalGb) ? ` · ${number(tiers.totalGb, 1)} GB` : ''}` : '—'}</span></div>
          <div className="smi-stack" role="img" aria-label={tr('Expert placement')}>{rows.map(r => <span key={r.id} data-tier={r.id} style={{ width: `${clamp(r.tier.pct, 100)}%` }}/>)}</div>
          {tiers ? <dl className="smi-memory__tiers">{rows.map(r => <div key={r.id} data-tier={r.id}><dt><i/>{tr(r.label)}</dt><dd className="num">{count(r.tier)}{hasReading(r.tier.gb) ? ` · ${number(r.tier.gb, 1)} GB` : ''} · {number(r.tier.pct, 1)}%</dd></div>)}</dl> : null}
          <p className="xs muted">{tr(tiers ? 'Experts in RAM were counted when the model loaded. Experts on disk are read on demand; the system file cache may hold some of them.' : 'This engine does not report where its experts are.')}</p>
        </div>
      </div>
    </section>;
}
```

### `apps/host/src/components/SmiMonitorExtras.tsx`

```tsx
/** SMI2000: the Monitor's extras - clear the recent requests, the sampling card, the KV format and the VRAM bar. */
import { useState } from 'react';
import { api } from '../lib/client';
import { localeTag, tr } from '../lib/i18n';
import { hasReading } from '../lib/monitor-hardware';
import { applyNativeEvent, type NativeState } from '../lib/native';
import { kvDetails, vramSegments } from '../lib/smi-expert-tiers';
import type { MonitorState } from '../lib/types';
import { Btn } from './ui';

const number = (value: number | null | undefined, digits = 0) => hasReading(value) ? value.toLocaleString(localeTag(), { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';

/** "Clear" in Recent requests: the first click asks, the second clears the list and the totals on the server (the model keeps running). */
export function SmiClearRequests({ ids }: { ids: string[] }) {
    const [armed, setArmed] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState('');
    if (!ids.length)
        return null;
    async function clear() {
        if (!armed) {
            setArmed(true);
            setTimeout(() => setArmed(false), 4000);
            return;
        }
        setBusy(true);
        setError('');
        try {
            for (const id of ids)
                applyNativeEvent(await api.post<NativeState>(`/api/native/instances/${encodeURIComponent(id)}/metrics/reset`, {}));
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setBusy(false);
            setArmed(false);
        }
    }
    return <span className="smi-clear">{error ? <span className="xs inline-error">{tr(error)}</span> : null}<Btn size="sm" disabled={busy} onClick={() => void clear()}>{tr(busy ? 'Clearing…' : armed ? 'Click again to clear' : 'Clear')}</Btn></span>;
}

const SAMPLING_ROWS: [string, string, number][] = [
    ['temperature', 'Temperature', 2], ['top_p', 'Top-p', 2], ['top_k', 'Top-k', 0], ['min_p', 'Min-p', 2],
    ['repetition_penalty', 'Repetition penalty', 2], ['presence_penalty', 'Presence penalty', 2], ['frequency_penalty', 'Frequency penalty', 2],
    ['penalty_last_n', 'Penalty window', 0], ['seed', 'Seed', 0],
];

/** The sampling a request gets for the fields it leaves out: the model card, then the model's config, then the Chat settings. */
export function SmiSamplingCard({ sampling }: { sampling: MonitorState['sampling'] }) {
    const thinking = sampling?.thinking ?? {}, instruct = sampling?.instruct ?? {}, config = sampling?.config ?? {};
    const rows = SAMPLING_ROWS.filter(([key]) => hasReading(thinking[key]) || hasReading(instruct[key]));
    const split = rows.some(([key]) => thinking[key] !== instruct[key]);
    const cell = (v: unknown, digits: number) => typeof v === 'number' ? number(v, digits) : '—';
    return <section className="st-card smi-sampling" aria-label={tr('Sampling')}>
      <h2 className="h4">{tr('Sampling')}</h2>
      {rows.length ? <table className="smi-sampling__table"><thead><tr><th/>{split ? <><th className="num">{tr('Thinking')}</th><th className="num">{tr('No thinking')}</th></> : <th className="num">{tr('Value')}</th>}</tr></thead>
        <tbody>{rows.map(([key, label, digits]) => <tr key={key}><td>{tr(label)}{key in config ? <span className="smi-sampling__mark" title={tr('Set in the model settings')}>●</span> : null}</td><td className="num">{cell(thinking[key], digits)}</td>{split ? <td className="num">{cell(instruct[key], digits)}</td> : null}</tr>)}</tbody></table>
        : <p className="small muted">{tr(sampling ? 'Greedy: no sampling values are set.' : 'Restart the model to see its sampling.')}</p>}
      <p className="xs muted">{tr('Used for the fields a request leaves out. ● set in the model settings; the others come from the model card or the Chat settings.')}</p>
    </section>;
}

/** The KV cache format under "Context fill": what it is and how much memory it takes. */
export function SmiKvInfo({ engine }: { engine: Record<string, any> | null | undefined }) {
    const kv = kvDetails(engine);
    if (!kv)
        return null;
    return <div className="smi-kv">
      <div className="monitor-bar-reading"><span>{tr('KV cache')}</span><span className="muted num">{kv.mode}{hasReading(kv.gb) ? ` · ≈${number(kv.gb, 2)} GB` : ''}</span></div>
      <p className="xs muted">{tr(kv.format)}{kv.note ? <> {tr(kv.note)}</> : null}{hasReading(kv.kbPerToken) ? <> {tr('About {0} KB per context token.', [number(kv.kbPerToken, 1)])}</> : null}{kv.streaming ? <> {tr('Kept in RAM; only the part attention reads is in VRAM.')}</> : null}</p>
    </div>;
}

/** One coloured bar of what fills the card: the engine's own start figures, the KV estimate, the rest and the free memory. */
export function SmiVramBar({ engine, usedGb, totalGb }: { engine: Record<string, any> | null | undefined; usedGb: number | null; totalGb: number | null | undefined }) {
    const segments = vramSegments(engine, usedGb, totalGb);
    if (!segments)
        return null;
    return <div className="smi-vram">
      <div className="smi-stack smi-stack--thin" role="img" aria-label={tr('VRAM use')}>{segments.map(s => <span key={s.id} data-part={s.id} style={{ width: `${s.pct}%` }} title={`${tr(s.label)}: ${number(s.gb, 2)} GB`}/>)}</div>
      <dl className="smi-vram__legend">{segments.map(s => <div key={s.id} data-part={s.id} title={tr(s.label)}><dt><i/>{tr(s.short)}</dt><dd className="num">{s.approx ? '≈' : ''}{number(s.gb, s.gb < 10 ? 2 : 1)}</dd></div>)}</dl>
    </div>;
}
```

### `apps/host/src/styles/smi-monitor.css`

```css
/* SMI2000: the Monitor's "Memory and experts" card (components/SmiMemoryPanel.tsx). */
.smi-memory { display: flex; flex-direction: column; gap: 12px; }
.smi-memory h2 { margin: 0; }
.smi-memory__grid { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1.6fr); gap: 16px 32px; }
.smi-memory__ram, .smi-memory__experts { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.smi-memory__ram .monitor-spark { margin-top: 8px; }
.smi-memory__facts { display: flex; flex-wrap: wrap; gap: 4px 16px; }
.smi-memory__experts p { margin: 0; }
.smi-stack { display: flex; height: 12px; border-radius: var(--st-r-pill); overflow: hidden; background: var(--st-surface-2); border: 1px solid var(--st-line-soft); }
.smi-stack > span { height: 100%; transition: width var(--st-dur) var(--st-ease); }
.smi-memory [data-tier='vram'] { --smi-tier: var(--st-accent); }
.smi-memory [data-tier='ram'] { --smi-tier: var(--st-info); }
.smi-memory [data-tier='disk'] { --smi-tier: var(--st-warn); }
.smi-stack > span { background: var(--smi-tier); }
.smi-memory__tiers { margin: 0; display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 8px 16px; }
.smi-memory__tiers > div { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.smi-memory__tiers dt { display: flex; align-items: center; gap: 6px; color: var(--st-ink-muted); font-size: 12px; }
.smi-memory__tiers dt i { width: 10px; height: 10px; border-radius: 3px; background: var(--smi-tier); flex: none; }
.smi-memory__tiers dd { margin: 0; font-size: 13px; font-weight: 600; overflow-wrap: anywhere; }
@media (max-width: 950px) { .smi-memory__grid { grid-template-columns: 1fr; } }
@media (max-width: 640px) { .smi-memory__tiers { grid-template-columns: 1fr; } }

/* SMI2000: the sampling card beside "Memory and experts" (the same columns as "Context fill" and "Recent requests"). */
.smi-row { display: grid; grid-template-columns: minmax(250px,1fr) minmax(0,2.4fr); gap: 16px; align-items: stretch; }
.smi-row[data-single='true'] { grid-template-columns: 1fr; }
.smi-sampling { display: flex; flex-direction: column; gap: 10px; }
.smi-sampling h2, .smi-sampling p { margin: 0; }
.smi-sampling__table { width: 100%; border-collapse: collapse; font-size: 12px; }
.smi-sampling__table th { font-size: 10px; font-weight: 500; color: var(--st-ink-muted); text-align: left; padding: 4px 0; }
.smi-sampling__table td { padding: 5px 0; border-top: 1px solid var(--st-line-soft); }
.smi-sampling__table .num { text-align: right; font-variant-numeric: tabular-nums; }
.smi-sampling__mark { margin-left: 6px; font-size: 8px; color: var(--st-accent); vertical-align: middle; }
@media (max-width: 950px) { .smi-row { grid-template-columns: 1fr; } }

/* SMI2000: the KV format under "Context fill". */
.smi-kv p { margin: 4px 0 0; line-height: 1.45; }

/* SMI2000: Recent requests - the Clear button, and room for the prefill column. */
.smi-toolbar-actions { display: flex; align-items: center; gap: 8px; margin-left: auto; }
.smi-clear { display: inline-flex; align-items: center; gap: 8px; }
.monitor-request-table .st-table { min-width: 700px; }

/* SMI2000: the VRAM card's bar of what fills the card. */
.smi-vram { display: flex; flex-direction: column; gap: 6px; }
.smi-stack--thin { height: 8px; }
.smi-vram [data-part='weights'] { --smi-tier: #8b5cf6; }
.smi-vram [data-part='draft'] { --smi-tier: #ec4899; }
.smi-vram [data-part='experts'] { --smi-tier: var(--st-accent); }
.smi-vram [data-part='kv'] { --smi-tier: var(--st-info); }
.smi-vram [data-part='buffers'] { --smi-tier: #14b8a6; }
.smi-vram [data-part='other'] { --smi-tier: var(--st-warn); }
.smi-vram [data-part='free'] { --smi-tier: transparent; }
.smi-vram .smi-stack > span { background: var(--smi-tier); }
.smi-vram__legend { margin: 0; display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 2px 10px; font-size: 10px; }
.smi-vram__legend > div { display: flex; justify-content: space-between; gap: 6px; min-width: 0; }
.smi-vram__legend dt { display: flex; align-items: center; gap: 4px; color: var(--st-ink-muted); min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.smi-vram__legend dt i { width: 8px; height: 8px; border-radius: 2px; background: var(--smi-tier); flex: none; }
.smi-vram [data-part='free'] dt i { border: 1px solid var(--st-line-soft); }
.smi-vram__legend dd { margin: 0; white-space: nowrap; }
```

### `apps/host/src/locales/smi-en.json`

```json
{
  "smi001": "Memory and experts",
  "smi002": "Expert placement",
  "smi005": "NVMe disk",
  "smi007": "used",
  "smi009": "Total",
  "smi010": "Experts kept in RAM",
  "smi011": "Experts in RAM were counted when the model loaded. Experts on disk are read on demand; the system file cache may hold some of them.",
  "smi012": "This engine does not report where its experts are.",
  "smi013": "Clear",
  "smi014": "Clearing…",
  "smi015": "Click again to clear",
  "smi016": "This Strata server cannot clear its requests (update the server).",
  "smi017": "The requests could not be cleared.",
  "smi018": "Prefill tok/s",
  "smi019": "Min-p",
  "smi020": "No thinking",
  "smi021": "Value",
  "smi022": "Set in the model settings",
  "smi023": "Greedy: no sampling values are set.",
  "smi024": "Restart the model to see its sampling.",
  "smi025": "Used for the fields a request leaves out. ● set in the model settings; the others come from the model card or the Chat settings.",
  "smi026": "KV cache",
  "smi027": "About {0} KB per context token.",
  "smi028": "Kept in RAM; only the part attention reads is in VRAM.",
  "smi029": "8-bit keys and values.",
  "smi030": "The default: precise.",
  "smi031": "4-bit keys and values, rotated before rounding.",
  "smi032": "Half the memory of 8-bit; less precise on long documents.",
  "smi033": "8-bit keys, 4-bit values.",
  "smi034": "23% less memory than 8-bit, so more experts fit in VRAM; prompts are 2-5% slower.",
  "smi035": "16-bit keys and values.",
  "smi036": "Twice the memory of 8-bit.",
  "smi037": "DeepSeek layout: the compressed cache in FP4, the recent window in FP8.",
  "smi038": "VRAM use",
  "smi039": "Draft model (MTP)",
  "smi040": "KV cache (context)",
  "smi041": "Verify buffers",
  "smi042": "Other (work memory, other programs)",
  "smi043": "Free",
  "smi044": "Weights",
  "smi045": "MTP draft",
  "smi046": "Experts",
  "smi047": "Buffers",
  "smi048": "Other",
  "smi049": "Copy to clipboard",
  "smi050": "Image copied",
  "smi051": "Paste it where you want with Ctrl+V.",
  "smi052": "The image was not copied",
  "smi053": "Use Download instead."
}
```

### `apps/host/src/locales/smi-es.json`

```json
{
  "smi001": "Memoria y expertos",
  "smi002": "Ubicación de los expertos",
  "smi005": "Disco NVMe",
  "smi007": "en uso",
  "smi009": "Total",
  "smi010": "Expertos guardados en RAM",
  "smi011": "Los expertos en RAM se contaron al cargar el modelo. Los expertos en disco se leen cuando hacen falta; la caché de archivos del sistema puede guardar algunos.",
  "smi012": "Este motor no indica dónde están sus expertos.",
  "smi013": "Borrar",
  "smi014": "Borrando…",
  "smi015": "Pulsa otra vez para borrar",
  "smi016": "Este servidor de Strata no puede borrar sus solicitudes (actualiza el servidor).",
  "smi017": "No se pudieron borrar las solicitudes.",
  "smi018": "Prefill tok/s",
  "smi019": "Min-p",
  "smi020": "Sin razonamiento",
  "smi021": "Valor",
  "smi022": "Definido en los ajustes del modelo",
  "smi023": "Greedy: no hay valores de muestreo definidos.",
  "smi024": "Reinicia el modelo para ver su muestreo.",
  "smi025": "Se usan para los campos que una solicitud no indica. ● definido en los ajustes del modelo; los demás vienen de la ficha del modelo o de los ajustes del chat.",
  "smi026": "Caché KV",
  "smi027": "Unos {0} KB por token de contexto.",
  "smi028": "Se guarda en RAM; en la VRAM solo está la parte que lee la atención.",
  "smi029": "Claves y valores de 8 bits.",
  "smi030": "El valor por defecto: preciso.",
  "smi031": "Claves y valores de 4 bits, rotados antes de redondear.",
  "smi032": "La mitad de memoria que 8 bits; menos preciso en documentos largos.",
  "smi033": "Claves de 8 bits, valores de 4 bits.",
  "smi034": "Un 23% menos de memoria que 8 bits, así caben más expertos en VRAM; los prompts son un 2-5% más lentos.",
  "smi035": "Claves y valores de 16 bits.",
  "smi036": "El doble de memoria que 8 bits.",
  "smi037": "Formato de DeepSeek: la caché comprimida en FP4, la ventana reciente en FP8.",
  "smi038": "Uso de la VRAM",
  "smi039": "Modelo borrador (MTP)",
  "smi040": "Caché KV (contexto)",
  "smi041": "Búferes de verificación",
  "smi042": "Otros (memoria de trabajo, otros programas)",
  "smi043": "Libre",
  "smi044": "Pesos",
  "smi045": "Borrador MTP",
  "smi046": "Expertos",
  "smi047": "Búferes",
  "smi048": "Otros",
  "smi049": "Copiar al portapapeles",
  "smi050": "Imagen copiada",
  "smi051": "Pégala donde quieras con Ctrl+V.",
  "smi052": "No se pudo copiar la imagen",
  "smi053": "Usa Descargar."
}
```

### `apps/host/tests/smi-monitor.test.mjs`

```js
// SMI2000: where the experts are (VRAM, RAM, NVMe) on the Monitor's "Memory and experts" card.
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { originalDashboard } from '../server/monitor-dashboard.mjs';

const built = await build({ stdin: { contents: "export { expertTiers, kvDetails, vramSegments } from './src/lib/smi-expert-tiers';", resolveDir: process.cwd(), loader: 'ts' }, bundle: true, write: false, format: 'esm', platform: 'node' });
const { expertTiers, kvDetails, vramSegments } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

test('Qwen: VRAM from the GPU cache, RAM from arena_mib, the rest on disk (measured on an RTX 3060 12 GB, 48 GB RAM, Q6_K_XL)', () => {
  const t = expertTiers({ used: 609, total: 24576, memoryGb: 2572 / 1024, totalGb: 104187 / 1024, ramGb: 24571 / 1024, source: 'engine' });
  assert.equal(t.vram.count, 609);
  assert.equal(t.ram.approx, true);
  assert.ok(Math.abs(t.ram.count - 5796) <= 2, String(t.ram.count));
  assert.equal(t.vram.count + t.ram.count + t.disk.count, 24576);
  assert.ok(Math.abs(t.vram.pct + t.ram.pct + t.disk.pct - 100) < 1e-6);
  assert.ok(Math.abs(t.disk.gb - (104187 - 2572 - 24571) / 1024) < 1e-6);
});

test('a full RAM copy never counts an expert twice', () => {
  const t = expertTiers({ used: 100, total: 1000, memoryGb: 1, totalGb: 10, ramGb: 10, source: 'engine' });
  assert.equal(t.disk.count, 0);
  assert.equal(t.ram.count, 900);
});

test('DeepSeek: counts from ram_experts, no sizes', () => {
  const t = expertTiers({ used: 300, total: 1000, ramUsed: 500, source: 'engine' });
  assert.deepEqual([t.vram.count, t.ram.count, t.disk.count], [300, 500, 200]);
  assert.equal(t.ram.approx, false);
  assert.equal(t.disk.pct, 20);
});

test('no total or no RAM figure: nothing is guessed', () => {
  assert.equal(expertTiers({ used: 609, total: NaN, memoryGb: 2.5, source: 'engine' }), null);
  assert.equal(expertTiers({ used: 609, total: 24576, memoryGb: 2.5, source: 'engine' }), null);
  assert.equal(expertTiers({ used: 0, total: 0, source: 'none' }), null);
});

test('the dashboard mapping carries the new engine facts', () => {
  const fallback = { context: { maxTokens: 4096 }, hardware: { gpus: [] }, totals: {}, telemetry: {} };
  const qwen = originalDashboard({ live: { state: 'idle' }, engine: { expert_slots: 609, expert_cache_mib: 2572, experts: 24576, expert_total_mib: 104187, arena_mib: 24571 } }, null, fallback).expertCache;
  assert.equal(qwen.total, 24576); assert.equal(qwen.totalGb, 104187 / 1024); assert.equal(qwen.ramGb, 24571 / 1024);
  const ds4 = originalDashboard({ live: { state: 'idle' }, engine: { expert_slots: 300, cpu_experts: 700, ram_experts: 500 } }, null, fallback).expertCache;
  assert.equal(ds4.total, 1000); assert.equal(ds4.ramUsed, 500);
});

test('the KV format: k8v4 with the server estimate, streaming said, an unknown mode named as it is', () => {
  const kv = kvDetails({ kv: 'k8v4', kv_mib_est: 820, context: 65536, kv_resident: 0 });
  assert.equal(kv.format, '8-bit keys, 4-bit values.');
  assert.ok(Math.abs(kv.gb - 820 / 1024) < 1e-9);
  assert.ok(kv.kbPerToken > 11 && kv.kbPerToken < 11.3, String(kv.kbPerToken));
  assert.equal(kv.streaming, false);
  assert.equal(kvDetails({ kv: 'int8', kv_resident: 32768 }).streaming, true);
  assert.equal(kvDetails({ kv: 'fp4' }).gb, null);
  assert.equal(kvDetails({ kv: 'zz9' }).format, 'zz9');
  assert.equal(kvDetails({}), null);
});

test('the VRAM bar: the engine parts, the KV estimate, the rest and the free memory fill the card exactly', () => {
  const engine = { vram_parts: { weights: 4480, draft: 1152, buffers: 114 }, expert_cache_mib: 2572, kv_mib_est: 820, context: 65536, kv_resident: 0 };
  const s = vramSegments(engine, 11670 / 1024, 12);
  assert.deepEqual(s.map(x => x.id), ['weights', 'draft', 'experts', 'kv', 'buffers', 'other', 'free']);
  assert.ok(Math.abs(s.reduce((n, x) => n + x.pct, 0) - 100) < 1e-9);
  assert.ok(Math.abs(s.find(x => x.id === 'other').gb - (11670 - 9138) / 1024) < 1e-9);
  assert.equal(s.find(x => x.id === 'kv').approx, true);
  const over = vramSegments(engine, 4, 12);   // parts above what the sensor reads: scaled down, never over 100%
  assert.ok(Math.abs(over.reduce((n, x) => n + x.pct, 0) - 100) < 1e-9);
  assert.equal(vramSegments({ kv_mib_est: 800 }, 8, 12), null);
  assert.equal(vramSegments(engine, null, 12), null);
});

test('the dashboard mapping passes the engine facts and the sampling on', () => {
  const fallback = { context: { maxTokens: 4096 }, hardware: { gpus: [] }, totals: {}, telemetry: {} };
  const sampling = { config: { temperature: 0.2 }, thinking: { temperature: 0.2 }, instruct: { temperature: 0.2 } };
  const out = originalDashboard({ live: { state: 'idle' }, engine: { kv: 'k8v4' }, sampling }, null, fallback);
  assert.equal(out.engineInfo.kv, 'k8v4');
  assert.deepEqual(out.sampling, sampling);
});

test('the host clears a loaded model\'s requests through its server and keeps the fresh figures', async () => {
  const { nativeFixture, finishJob } = await import('./helpers/native-fixture.mjs');
  const f = nativeFixture();
  try {
    await f.control.refresh();
    const done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(done.state, 'done', done.error);
    const state = await f.control.resetMetrics(done.result.instanceId);
    const instance = state.instances.find(i => i.id === done.result.instanceId);
    assert.equal(instance.metrics.engine.max_context, 32768);
    assert.equal('key' in instance, false);
    await assert.rejects(() => f.control.resetMetrics('no-such-model'), /loaded model/);
  } finally { await f.cleanup(); }
});
```

### `serve/test_smi_monitor.py`

```python
"""SMI2000: the Monitor's extras on GET /metrics - POST /metrics/reset clears the requests and the totals of the /metrics
view only, the default sampling is reported, the pack's expert totals and the VRAM parts the engine wrote at start."""
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, Service, kv_estimate_mib, pack_expert_totals, serve,
                          vram_parts_from_log)
from serve.test_lifecycle import ResidentEngine

START_LOG = """strata generate: 216 MiB of weights loaded from C:\\packs\\q6 in 0.1 s
strata: STRATA_HC_Q8=1: 0.63 GiB of Q8_0 hyper-connection projections for the verify read
strata generate: 499 native projection matrices, 3619.14 MiB of weights, in 3.2 s
strata mtp: draft layer loaded, 876 MiB of VRAM (experts 675, dense 111), files read in 0.36 s (2197 MiB/s)
strata verify: window up to 6 tokens, 113.8 MiB of device buffers
strata mtp: draft head over 106299 tokens (275.7 MiB)
"""


class Helpers(unittest.TestCase):
    def test_vram_parts(self):
        self.assertEqual(vram_parts_from_log(START_LOG), {"weights": 4480, "draft": 1152, "buffers": 114})
        self.assertEqual(vram_parts_from_log("nothing here"), {})

    def test_kv_estimate(self):
        self.assertEqual(kv_estimate_mib("qwen4exp", "k8v4", 65536), 820)
        self.assertIsNone(kv_estimate_mib("deepseek41", "fp4", 65536))
        self.assertIsNone(kv_estimate_mib("qwen4exp", "k8v4", None))

    def test_pack_totals(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "native_experts.txt").write_text(
                "# strata native experts v4: layer ... (n_expert 512, total 1073741824; absolute offsets)\n0 1\n1 2\n",
                encoding="utf-8")
            self.assertEqual(pack_expert_totals(["--pack", d]), {"experts": 1024, "expert_total_mib": 1024})
        self.assertEqual(pack_expert_totals(["--pack", "/no/such/pack"]), {})
        self.assertEqual(pack_expert_totals([]), {})


class Route(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.engine.loaded = True
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.sampling_defaults = {"temperature": 0.2, "top_k": 1}
        self.svc.model_arch = "qwen4exp"
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode())

    def test_sampling_reported(self):
        body = self.request("/metrics")[1]
        self.assertEqual(body["sampling"]["config"], {"temperature": 0.2, "top_k": 1})
        self.assertEqual(body["sampling"]["thinking"]["temperature"], 0.2)
        self.assertEqual(body["sampling"]["instruct"]["top_k"], 1)

    def test_reset_clears_the_metrics_view_only(self):
        with self.svc.status_lock:
            self.svc.history.append({"time": 1.0, "prompt_tokens": 10, "output_tokens": 5})
            self.svc.totals.update(requests=1, prompt_tokens=10, output_tokens=5)
        self.assertEqual(len(self.request("/metrics")[1]["requests"]), 1)
        code, body = self.request("/metrics/reset", {})
        self.assertEqual((code, body), (200, {"status": "cleared"}))
        after = self.request("/metrics")[1]
        self.assertEqual(after["requests"], [])
        self.assertEqual((after["totals"]["requests"], after["totals"]["output_tokens"]), (0, 0))
        self.assertEqual(self.svc.totals["requests"], 1)               # /v1/status and Prometheus keep counting
        with self.svc.status_lock:
            self.svc.totals.update(requests=2, output_tokens=12)
        self.assertEqual(self.request("/metrics")[1]["totals"]["output_tokens"], 7)

    def test_reset_wants_json_from_the_own_page(self):
        req = urllib.request.Request(self.base + "/metrics/reset", data=b"x=1",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(caught.exception.code, 415)


if __name__ == "__main__":
    unittest.main()
```
