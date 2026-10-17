export type Window = {start: number; end: number};

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

export function mergeWindows(windows: Window[], gap = 0): Window[] {
  if (!Number.isFinite(gap) || gap < 0) {
    throw new RangeError(`gap must be a finite nonnegative number, got ${gap}`);
  }
  if (!Array.isArray(windows)) {
    throw new RangeError('windows must be an array');
  }
  for (const w of windows) {
    if (
      w == null ||
      !isFiniteNumber(w.start) ||
      !isFiniteNumber(w.end) ||
      w.start > w.end
    ) {
      throw new RangeError(
        `invalid window: start and end must be finite numbers with start <= end, got ${JSON.stringify(w)}`,
      );
    }
  }

  const sorted = [...windows].sort(
    (a, b) => a.start - b.start || a.end - b.end,
  );

  const result: Window[] = [];
  for (const w of sorted) {
    const last = result.at(-1);
    if (last !== undefined && w.start - last.end <= gap) {
      if (w.end > last.end) last.end = w.end;
    } else {
      result.push({start: w.start, end: w.end});
    }
  }
  return result;
}
