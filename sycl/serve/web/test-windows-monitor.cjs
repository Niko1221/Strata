// Run with node sycl/serve/web/test-windows-monitor.cjs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const cards = {};
let calls = 0;
const scope = {renderMonitor: () => calls++, fmt: x => String(x),
               setMetric: (key, value, unit, sub) => { cards[key] = {value, unit, sub}; }};
vm.createContext(scope);
vm.runInContext(fs.readFileSync(__dirname + '/windows-monitor.js', 'utf8'), scope);
scope.renderMonitor({}, {gpu_backend: 'igcl', gpu_power: 0, gpu_temp: 56, gpu_vram_temp: 58,
                        gpu_pcie_gen: 3, gpu_pcie_gen_max: 5, gpu_pcie_width: 16});
assert.equal(cards.power.value, '0');
assert.equal(cards.power.sub, 'GPU chip power (Intel)');
assert.equal(cards.temp.sub, 'VRAM 58 °C');
assert.equal(cards.pcie.value, 'Gen3');
assert.equal(cards.pcie.unit, 'x16');
assert.match(cards.pcie.sub, /Gen5.*Traffic unavailable/);
scope.renderMonitor({}, {gpu_backend: 'igcl'});
assert.equal(cards.power.value, null);
assert.equal(cards.pcie.value, null);
const before = JSON.stringify(cards);
scope.renderMonitor({}, {gpu_power: 120});
assert.equal(JSON.stringify(cards), before);
assert.equal(calls, 3);
console.log('Windows Monitor labels: passed');
