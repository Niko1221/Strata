// strata-mi50 (2026-10-26): the Monitor's "Record log" button.  While on, the Monitor's readings (phase, speeds, CPU load, RAM, disk read
// and write, every GPU's load / VRAM / PCIe / power / temperature) are sampled once a second; Stop saves them with the request list and
// the hardware inventory as Strata-Diagnostics-<time>.json in Downloads.  The engine's own per-request lines stay in its log
// (strata-<model>.log); the file's start and end times place a recording against it.
// The recorder lives here, outside the Monitor screen, so a recording goes on while another screen (a chat) is open (first version:
// inside the Monitor, and leaving it stopped the recording).
import { useSyncExternalStore } from 'react';
import { nativeState, refreshNative } from './native';
import { nativeMonitorData, withHostSensors } from './native-monitor';
import { getState } from './store';
import type { MonitorState } from './types';

const SAMPLE_MS = 1000;
const MAX_SAMPLES = 3600;   // an hour at one a second

type Sample = {
    t: string;
    phase: string;
    outputTps: number;
    promptTps: number;
    cpuPct: number | null;
    ramUsedGb: number | null;
    diskReadMbs: number | null;
    diskWriteMbs: number | null;
    gpus: { id: string; name: string; loadPct: number | null; vramGb: number | null; pcieRxMbs: number | null; powerW: number | null; tempC: number | null }[];
    prompt: unknown;
    context: unknown;
    expertCache: unknown;
    hardware: unknown;
};

const num = (v: unknown) => typeof v === 'number' && Number.isFinite(v) ? v : null;

/** The readings the Monitor would show for all models and all GPUs, from the app's shared state (no screen needed). */
function current(): MonitorState | null {
    const app = getState();
    const native = nativeState();
    if (native?.instances.length)
        return withHostSensors(nativeMonitorData(native, app.hardware, 'all'), app.monitor);
    return app.monitor;
}

function sample(data: MonitorState): Sample {
    const h = data.hardware as Record<string, any>;
    const read = num(h.diskReadGbs);
    const names = new Map((getState().hardware?.gpus ?? []).map(g => [String(g.id ?? ''), g.name]));
    return {
        t: new Date().toISOString(),
        phase: data.phase,
        outputTps: data.outputTps,
        promptTps: data.promptTps,
        cpuPct: num(h.cpuLoad),
        ramUsedGb: num(h.ramUsedGb),
        diskReadMbs: read === null ? null : read * 1024,
        diskWriteMbs: num(h.diskWriteMbs),
        gpus: (h.gpus ?? []).map((g: Record<string, any>) => ({
            id: String(g.id ?? ''), name: String(g.name ?? names.get(String(g.id ?? '')) ?? ''), loadPct: num(g.utilizationPct),
            vramGb: num(g.memoryUsedGb), pcieRxMbs: num(g.pcieRxMbs), powerW: num(g.powerW), tempC: num(g.temperatureC),
        })),
        prompt: data.prompt,
        context: data.context,
        expertCache: data.expertCache,
        hardware: data.hardware,
    };
}

function download(name: string, text: string) {
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

let samples: Sample[] = [];
let startedAt = '';
let timer: ReturnType<typeof setInterval> | null = null;
let snapshot = { recording: false, count: 0 };
const listeners = new Set<() => void>();
const publish = () => { snapshot = { recording: timer !== null, count: samples.length }; listeners.forEach(fn => fn()); };

export function startRecording() {
    if (timer)
        return;
    samples = [];
    startedAt = new Date().toISOString();
    timer = setInterval(() => {
        void refreshNative();   // the native state is polled only while a screen uses it: keep it fresh for the recording
        const data = current();
        if (data && samples.length < MAX_SAMPLES) {
            samples.push(sample(data));
            publish();
        }
    }, SAMPLE_MS);
    publish();
}

/** Stops and saves; returns the file name. */
export function stopRecording(): string {
    if (timer)
        clearInterval(timer);
    timer = null;
    const data = current();
    const name = `Strata-Diagnostics-${startedAt.slice(0, 19).replace(/[:T]/g, '-')}.json`;
    const file = {
        schema: 'strata-mi50-diagnostics/2',
        startedAt,
        endedAt: new Date().toISOString(),
        sampleMs: SAMPLE_MS,
        inventory: getState().hardware,
        requests: data?.requests ?? [],
        totals: data?.totals ?? null,
        samples,
    };
    download(name, JSON.stringify(file, null, 1));
    publish();
    return name;
}

export function useDiagnosticsRecording() {
    return useSyncExternalStore(fn => { listeners.add(fn); return () => listeners.delete(fn); }, () => snapshot);
}
