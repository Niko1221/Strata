export type Window = {start: number; end: number};

function assertValidWindow(value: Window): void {
  const {start, end} = value ?? ({} as Window);
  if (
    typeof start !== 'number' ||
    typeof end !== 'number' ||
    !Number.isFinite(start) ||
    !Number.isFinite(end) ||
    start > end
  ) {
    throw new RangeError(`invalid window: ${JSON.stringify(value)}`);
  }
}

export function mergeWindows(windows: Window[], gap = 0): Window[] {
  if (typeof gap !== 'number' || !Number.isFinite(gap) || gap < 0) {
    throw new RangeError(`invalid gap: ${String(gap)}`);
  }
  for (const window of windows) assertValidWindow(window);

  const sorted = [...windows].sort(
    (a, b) => a.start - b.start || a.end - b.end,
  );

  const result: Window[] = [];
  for (const window of sorted) {
    const last = result[result.length - 1];
    if (last && window.start <= last.end + gap) {
      if (window.end > last.end) last.end = window.end;
    } else {
      result.push({start: window.start, end: window.end});
    }
  }
  return result;
}
