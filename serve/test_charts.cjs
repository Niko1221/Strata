// Run with node serve/test_charts.cjs. Exercise the actual dashboard functions without a browser/GPU.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/web/app.js', 'utf8');
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, {textContent:'', innerHTML:'', attrs:{},
    setAttribute(k, v) { this.attrs[k] = v; }, querySelector(s) { return element(id + s); }});
  return elements.get(id);
};
const ctx = vm.createContext({document:{getElementById:element}});
vm.runInContext(source.slice(source.indexOf('const $ ='), source.indexOf('const store =')) +
  source.slice(source.indexOf('function spark('), source.indexOf('let lastMetrics =')), ctx);
vm.runInContext(`
  appleSpark('test', [null, null]);
`, ctx);
assert.equal(element('test.line').attrs.d, '');
vm.runInContext(`appleSpark('test', [0, 0]);`, ctx);
assert.equal(element('test.line').attrs.d, 'M0.00,30.00L100.00,30.00');
vm.runInContext(`appleSpark('test', [1, 2, null, 3, 4], 4);`, ctx);
assert.equal(element('test.line').attrs.d, 'M0.00,23.50L25.00,17.00M75.00,10.50L100.00,4.00');
assert.equal((element('test.area').attrs.d.match(/Z/g) || []).length, 2);
vm.runInContext(`appleSpark('test', [NaN, Infinity, undefined, 1]);`, ctx);
assert.equal(element('test.line').attrs.d, '');

vm.runInContext(`renderHardware(
  {gpu_util:62, gpu_mem_used:12884901888, gpu_mem_total:38654705664,
   gpu_temp:52, gpu_power:7.25, gpu_freq_mhz:924, cpu:35, disk_read_mb:740},
  {gpu_backend:'apple', gpu_sensor_helper:true, gpu_name:'Apple M3 Pro'},
  {gpu_freq_mhz:[800,924]}, {});`, ctx);
assert.equal(element('ml-vram').textContent, 'GPU allocation');
assert.equal(element('ml-pcie').textContent, 'GPU frequency');
assert.match(element('mv-pcie').innerHTML, /924.*MHz/);
assert.match(element('ms-vram').textContent, /shared unified memory/);
assert.notEqual(element('sp-pcie.line').attrs.d, '');

vm.runInContext(`renderHardware({}, {gpu_backend:'apple', gpu_sensor_helper:false}, {}, {});`, ctx);
assert.equal(element('mv-temp').innerHTML, '–');
assert.match(element('ms-temp').textContent, /Install macmon/);
assert.equal(element('sp-temp.line').attrs.d, '');

vm.runInContext(`renderHardware({gpu_pcie_gen:4, gpu_pcie_width:16, gpu_pcie_rx_mb:500}, {},
  {gpu_pcie_rx_mb:[400,500]}, {});`, ctx);
assert.equal(element('ml-pcie').textContent, 'PCIe');
assert.equal(element('ml-vram').textContent, 'VRAM');
assert.equal(element('ml-power').textContent, 'Power');
assert.match(element('mv-pcie').innerHTML, /Gen4.*x16/);
vm.runInContext(`spark('legacy', [null, null]);`, ctx);
assert.equal(element('legacy.line').attrs.d, 'M0.00,30.00L100.00,30.00');
vm.runInContext(`renderHardware({}, {gpu_note:'AMD on Windows'}, {}, {});`, ctx);
assert.equal(element('ms-gpu').textContent, 'not available');
assert.equal(element('ms-vram').textContent, 'not available on Windows AMD yet');
const appleCache = vm.runInContext(`expertCache({backend:'mlx', expert_memory_gb:'8.0'}, true)`, ctx);
assert.equal(appleCache.bytes, 8000000000);
assert.equal(appleCache.label, 'Expert cache budget');
const pcCache = vm.runInContext(`expertCache({expert_cache_mib:4096, expert_slots:123}, false)`, ctx);
assert.equal(pcCache.bytes, 4294967296);
assert.match(pcCache.text, /123.*4/);
console.log('Chart tests passed: gaps, zero, Apple sensors, missing helper, NVIDIA/AMD labels.');
