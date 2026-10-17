// Loaded only by the Windows Intel wrapper; the shared web app is unchanged.
const renderIntelMonitorBase = renderMonitor;
renderMonitor = function(live, hw, st, eng, h, last, requests, totals, kept) {
  renderIntelMonitorBase(live, hw, st, eng, h, last, requests, totals, kept);
  if (hw.gpu_backend !== "igcl") return;
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", "GPU chip power (Intel)");
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            hw.gpu_vram_temp == null ? "" : `VRAM ${fmt(hw.gpu_vram_temp)} °C`);
  setMetric("pcie", hw.gpu_pcie_gen ? `Gen${hw.gpu_pcie_gen}` : null,
            hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            (hw.gpu_pcie_gen_max ? `GPU supports Gen${hw.gpu_pcie_gen_max} · ` : "") + "Traffic unavailable");
};
