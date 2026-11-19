/** Physical-memory containment for an owned Windows GGUF engine, including load. */
import { freemem } from 'node:os';
export const GiB = 1024 ** 3;
export const RAM_FLOOR_BYTES = 3 * GiB;
// Leave time for cancellation and process shutdown, rather than reacting at the floor.
export const RAM_STOP_BYTES = 5 * GiB;

export function admitLocalMemory(availableBytes, { fullyOffloaded = false, modelBytes = 0, mapped = true } = {}) {
  // Fully GPU-resident weights can still occupy staging RAM and mapped pages.
  // This conservative load allowance is not used for disk-streamed large models.
  const staging = fullyOffloaded && Number.isFinite(modelBytes) && modelBytes > 0 ? modelBytes * (mapped ? 2 : 1) : 0;
  const required = RAM_STOP_BYTES + staging;
  if (!Number.isFinite(availableBytes) || availableBytes <= required)
    throw new Error(`RAM protection: this load needs more than ${(required / GiB).toFixed(1)} GiB of available system memory, including its staging allowance. Close applications or disable memory mapping, then try again. Context and user settings were not changed.`);
}

/** One guard owns one child. It never follows a replacement or retries a stop. */
export function createLocalMemoryGuard({ sample = freemem, now = Date.now, onPressure, setTimer = setInterval, clearTimer = clearInterval } = {}) {
  let timer = null, stopped = false, previous = null, minimum = Infinity;
  const state = { samples: 0, minimumAvailableBytes: null, pressure: null, stopError: null };
  const dispose = () => { stopped = true; if (timer !== null) clearTimer(timer); timer = null; };
  const check = async () => {
    if (stopped) return;
    let available;
    try { available = sample(); } catch { available = NaN; }
    const at = now();
    state.samples++;
    if (Number.isFinite(available) && available >= 0) { minimum = Math.min(minimum, available); state.minimumAvailableBytes = minimum; }
    const fallingPerMs = previous && at > previous.at ? Math.max(0, (previous.available - available) / (at - previous.at)) : 0;
    const threshold = Math.min(8 * GiB, Math.max(RAM_STOP_BYTES, RAM_FLOOR_BYTES + fallingPerMs * 5000));
    previous = { at, available };
    if (Number.isFinite(available) && available > threshold) return;
    state.pressure = { availableBytes: Number.isFinite(available) ? available : null, thresholdBytes: threshold, floorBytes: RAM_FLOOR_BYTES };
    dispose();
    try { await onPressure?.(state.pressure); } catch (error) { state.stopError = String(error?.message ?? error); }
  };
  return {
    start() { if (timer !== null || stopped) return; timer = setTimer(() => void check(), 250); timer.unref?.(); void check(); },
    check, dispose, snapshot: () => structuredClone(state),
  };
}
