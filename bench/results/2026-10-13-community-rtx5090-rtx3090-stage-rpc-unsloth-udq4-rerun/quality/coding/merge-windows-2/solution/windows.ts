export type Window = {start: number; end: number};

function assertFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be a finite number, got ${String(value)}`);
  }
  return value;
}

export function mergeWindows(windows: Window[], gap = 0): Window[] {
  const g = assertFiniteNumber(gap, 'gap');
  if (g < 0) {
    throw new RangeError(`gap must be nonnegative, got ${String(gap)}`);
  }
  if (!Array.isArray(windows)) {
    throw new TypeError('windows must be an array');
  }

  const normalized: Window[] = windows.map((window, index) => {
    if (window === null || typeof window !== 'object') {
      throw new RangeError(`windows[${index}] must be an object`);
    }
    const start = assertFiniteNumber(window.start, `windows[${index}].start`);
    const end = assertFiniteNumber(window.end, `windows[${index}].end`);
    if (start > end) {
      throw new RangeError(
        `windows[${index}] start (${start}) must be <= end (${end})`,
      );
    }
    return {start, end};
  });

  normalized.sort((a, b) => a.start - b.start || a.end - b.end);

  const result: Window[] = [];
  for (const window of normalized) {
    const last = result[result.length - 1];
    if (last !== undefined && window.start <= last.end + g) {
      if (window.end > last.end) last.end = window.end;
    } else {
      result.push({start: window.start, end: window.end});
    }
  }
  return result;
}
