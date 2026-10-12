export type Window = {start: number; end: number};

function assertFiniteNumber(value: unknown, label: string): void {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be a finite number, got ${String(value)}`);
  }
}

export function mergeWindows(windows: Window[], gap = 0): Window[] {
  assertFiniteNumber(gap, 'gap');
  if (gap < 0) throw new RangeError(`gap must be nonnegative, got ${gap}`);

  for (const window of windows) {
    if (window === null || typeof window !== 'object') {
      throw new RangeError(`window must be an object, got ${String(window)}`);
    }
    assertFiniteNumber(window.start, 'window.start');
    assertFiniteNumber(window.end, 'window.end');
    if (window.start > window.end) {
      throw new RangeError(
        `window.start must be <= window.end, got ${window.start} > ${window.end}`,
      );
    }
  }

  const sorted = [...windows].sort((a, b) => a.start - b.start || a.end - b.end);
  const result: Window[] = [];
  for (const window of sorted) {
    const last = result[result.length - 1];
    if (last && window.start - last.end <= gap) {
      last.end = Math.max(last.end, window.end);
    } else {
      result.push({start: window.start, end: window.end});
    }
  }
  return result;
}
