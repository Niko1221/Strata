/** Anonymous UI example only. Binds loopback; every mutation is unavailable. */
import {createServer} from 'node:http';
import {createHash} from 'node:crypto';
import {extname, join, resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {regularBytes, verifyCandidate} from '../assets.mjs';

const {values} = parseArgs({options:{candidate:{type:'string'}, 'manifest-sha256':{type:'string'}, port:{type:'string',default:'18128'}}});
if (!values.candidate || !values['manifest-sha256']) throw Error('candidate_and_pin_required');
const root = resolve(values.candidate), manifest = JSON.parse(await regularBytes(join(root, 'manifest.json'), 2 * 1024 * 1024));
// This verifies frozen example bytes, not the current installed App or a model.
await verifyCandidate(root, {manifestSha256:values['manifest-sha256'], current:manifest.capabilities});
const catalog = structuredClone(manifest.capabilities.catalog);
catalog.this_pc = {ram_gb:64, engine_vendor:'nvidia', gpus:[{name:'Example GPU', vendor:'nvidia', vram_gb:16, usable:true}]};
catalog.summary = 'Anonymous UI Fixture. No Model Backend.';
if (catalog.recommendation) {
  catalog.recommendation.why = 'Synthetic Hardware For This Anonymous UI Example.';
  catalog.recommendation.gpu = {...catalog.this_pc.gpus[0], index:0};
}
const model = {id:'example-local-model', name:'Example Local Model', contextMax:204800,
  reasoning:true, thinks:true, vision:false, reads:false};
const providers = {catalog:[], connections:[{id:'example-local-server', kind:'custom', provider:null,
  name:'Example Local Server', baseUrl:'/v1', protocol:'openai-chat', apiKeySet:false, models:[model]}]};
const native = {available:true, version:manifest.app.controlVersion, repo:'',
  summary:'Anonymous UI Fixture. No Model Backend.', error:null, catalog, installed:null,
  hardware:catalog.this_pc, fields:[], instances:[], defaultId:null, jobs:[], visionReader:false,
  engine:{usable:true, notice:'Anonymous UI Fixture Only.', version:null},
  capabilities:{responses:false, updates:false, rollback:false, calibration:false, componentUpdates:false}};
const routes = {
  '/api/setup':{complete:true, dismissed:true, step:0, history:[], app:{version:manifest.app.version, modelsDir:'', dataDir:''}},
  '/api/native':native, '/api/model-providers':providers, '/api/tasks':{tasks:[]},
  '/api/hardware':null, '/api/engine':null, '/api/serving':null, '/api/monitor':null, '/api/runtime':null,
  '/api/skills':[], '/api/workspaces':{workspaces:[]}, '/api/workspace':{path:'', entries:[]}, '/api/fonts':[],
  '/api/app/mode':{mode:'web', backgroundOnly:false}, '/api/agent/status':{state:'unavailable'},
  '/api/agent/pending':{pending:[]},
  '/api/models':{mode:'real', current:null, loaded:null, variants:[],
    downloads:{active:0, queued:0, totalGbToday:null}, disk:{modelsDir:'', usedGb:null, freeGb:null}},
  '/api/image-models':{mode:'real', family:'', sourceUrl:'', revision:'', selected:'off', applied:'off', generationAvailable:false, variants:[]},
};
const files = new Map(manifest.files.map(file => ['/' + file.path, file]));
const mime = {'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8', '.css':'text/css',
  '.svg':'image/svg+xml','.png':'image/png','.ico':'image/x-icon','.woff2':'font/woff2','.woff':'font/woff',
  '.json':'application/json','.webmanifest':'application/manifest+json'};
const port = Number(values.port);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw Error('invalid_fixture_port');
const server = createServer(async (req, res) => {
  const json = (status, value) => {res.writeHead(status, {'Content-Type':'application/json', 'Cache-Control':'no-store'});res.end(JSON.stringify(value));};
  try {
    if (!['GET','HEAD'].includes(req.method)) return json(501, {error:'Anonymous UI Fixture. No Backend Is Started.'});
    const url = new URL(req.url, 'http://127.0.0.1'), path = url.pathname;
    if (path === '/api/events') {
      res.writeHead(200, {'Content-Type':'text/event-stream', 'Cache-Control':'no-store'});
      res.write('event: providers\ndata: ' + JSON.stringify(providers) + '\n\n');
      res.write('event: native\ndata: ' + JSON.stringify(native) + '\n\n');
      return;
    }
    if (Object.hasOwn(routes, path)) return json(200, routes[path]);
    const file = files.get(path === '/' ? '/index.html' : path);
    if (!file) return json(404, {error:'Fixture Route Unavailable.'});
    const bytes = await regularBytes(join(root, 'dist', file.path));
    if (bytes.length !== file.size || createHash('sha256').update(bytes).digest('hex') !== file.sha256)
      throw Error('fixture_asset_changed');
    res.writeHead(200, {'Content-Type':mime[extname(file.path)] || 'application/octet-stream', 'Cache-Control':'no-store'});
    res.end(req.method === 'HEAD' ? undefined : bytes);
  } catch { if (!res.headersSent) json(503, {error:'Fixture Assets Unavailable.'});else res.end(); }
});
server.listen(port, '127.0.0.1', () => process.stdout.write(JSON.stringify({state:'anonymous-ui-fixture',
  url:'http://127.0.0.1:' + port, inferenceAvailable:false, installedAppVerified:false}) + '\n'));
