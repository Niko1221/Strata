/**
 * GPU energy since the host started, in Wh (#3628): the power reading held for the time since the last tick, added up.
 * A long gap (the PC slept, the host was paused) counts for at most 5 s, so a sleep never adds a made-up hour of use.
 */
const meter = { wh: 0, at: 0 };

export function addEnergy(powerW, at) {
  const dt = meter.at ? Math.min((at - meter.at) / 1000, 5) : 0;
  meter.at = at;
  if (typeof powerW === 'number' && Number.isFinite(powerW) && powerW >= 0 && dt > 0) meter.wh += powerW * dt / 3600;
  return meter.wh;
}

export const energyWh = () => meter.wh;
