/* Top app-bar status precision only. Keep calculations and other screens at their original precision. */
"use strict";
window.StrataMetrics = (() => {
  const precision = {minimumFractionDigits: 4, maximumFractionDigits: 4};
  const valid = value => value != null && Number.isFinite(Number(value));
  const number = value => valid(value) ? new Intl.NumberFormat(undefined, precision).format(Number(value)) : "–";
  const compact = value => valid(value) ? new Intl.NumberFormat(undefined,
    {...precision, notation: Number(value) >= 10000 ? "compact" : "standard"}).format(Number(value)) : "–";
  const money = (value, currency = "USD") => valid(value) ? new Intl.NumberFormat(undefined,
    {...precision, style: "currency", currency}).format(Number(value)) : "Set rate";
  const duration = value => {
    if (!valid(value)) return "Not recorded";
    const total = Math.max(0, Math.floor(Number(value)));
    const hours = Math.floor(total / 3600), minutes = Math.floor(total / 60) % 60;
    const seconds = String(total % 60);
    return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${seconds.padStart(2, "0")}` :
      minutes ? `${minutes}m ${seconds.padStart(2, "0")}s` : `${seconds}s`;
  };
  return {number, compact, money, duration};
})();
