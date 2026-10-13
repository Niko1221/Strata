export type Window = {start: number; end: number};

export function mergeWindows(windows: Window[], gap = 0): Window[] {
  if (!Number.isFinite(gap) || gap < 0) {
    throw new RangeError("gap must be a finite nonnegative number");
  }
  for (const w of windows) {
    if (!Number.isFinite(w.start) || !Number.isFinite(w.end) || w.start > w.end) {
      throw new RangeError("window must have finite start <= end");
    }
  }
  const sorted = windows.map(({start, end}) => ({start, end}));
  sorted.sort((a, b) => a.start - b.start || a.end - b.end);
  const result: Window[] = [];
  for (const w of sorted) {
    const last = result[result.length - 1];
    if (last && w.start - last.end <= gap) {
      if (w.end > last.end) last.end = w.end;
    } else {
      result.push(w);
    }
  }
  return result;
}
