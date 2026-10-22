/** Reviewed App 0.2.1/0.2.2 source anchors, not a router, permission list or live probe. */
export const REVIEWED_APP_VERSIONS = Object.freeze(['0.2.1', '0.2.2']);
// Whitespace changes are harmless; changed route/method/owner anchors need review.
const compact = text => text.replace(/\s+/g, '');
const route = (path, method, ownerCall) => ({
  api: { path, method },
  anchors: [`path === '${path}' && method === '${method}'`, ownerCall],
});
const read = (path, ownerCall) => ({
  api: { path, method: 'GET' },
  // These handlers have no method guard. GET describes the reviewed read usage,
  // not a claim that the Host rejects other methods.
  anchors: [`if (path === '${path}')`, ownerCall],
});
const nativeInstance = action => ({
  api: { path: `/api/native/instances/:id/${action}`, method: 'POST' },
  anchors: [String.raw`const nativeInstance=/^\/api\/native\/instances\/([a-z0-9-]+)\/(unload|default)$/.exec(path);`,
    "if(nativeInstance&&method==='POST')", action === 'unload' ? 'native.unload(nativeInstance[1])' : 'native.setDefault(nativeInstance[1])'],
});
const nativeSettings = method => ({
  api: { path: '/api/native/models/:id/settings', method },
  anchors: [String.raw`const nativeSettings=/^\/api\/native\/models\/([a-z0-9_-]+)\/settings$/.exec(path);`,
    `if(nativeSettings&&method==='${method}')`, method === 'GET' ? 'native.settings(nativeSettings[1])' : 'native.save(nativeSettings[1],await readBody(req))'],
});
const chat = () => ({
  api: { path: '/api/chat/stream', method: 'POST' },
  anchors: ["if ((path === '/api/chat/stream' || path === '/api/plan/start-goal') && method === 'POST')", 'runAgentTurn('],
});
const reviewed = [
  ['models', 'official-app-control', [route('/api/native', 'GET', 'native.refresh('),
    route('/api/native/plan', 'POST', 'native.plan(')]],
  ['monitor', 'official-host', [read('/api/monitor', 'monitor.snapshot()'),
    route('/api/serving', 'GET', 'serving.state()'), read('/api/events', 'hub.add(res)')]],
  ['providers', 'official-host', [route('/api/model-providers', 'GET', 'modelProviders.list()'),
    route('/api/model-providers/discover', 'POST', 'modelProviders.discover('),
    route('/api/model-providers', 'POST', 'modelProviders.save(')]],
  ['lifecycle', 'official-app-control', [route('/api/native/operations', 'POST', 'native.operation(body.kind,body.args)'),
    nativeInstance('unload'), nativeInstance('default')]],
  ['context-idle', 'official-app-control', [nativeSettings('GET'), nativeSettings('PATCH')]],
  ['chat', 'official-harness', [chat(), route('/api/chat/stop', 'POST', 'agent.stop(b.sessionId)'),
    route('/api/agent/pending', 'GET', 'agent.pendingFor('),
    route('/api/chat/approval', 'POST', 'agent.approve(')]],
  ['tools-search', 'official-harness', [route('/api/agent/tools', 'GET', 'agentTools.view()'),
    route('/api/agent/tools', 'PUT', 'agentTools.update(await readBody(req))')]],
  ['research', 'official-harness', [{ ...chat(), anchors: [...chat().anchors,
    String.raw`const researchLine = /^\/research(?:\s+([\s\S]*))?$/i.exec(message.trim());`, 'agent.researchCommand('] }]],
  ['skills', 'official-host', [route('/api/skills', 'GET', 'syncAgentSkills()'),
    route('/api/skills', 'POST', 'writeCreatedSkill(')]],
  ['tasks', 'official-host', [route('/api/tasks', 'GET', 'tasks.list()'), route('/api/tasks', 'POST', 'tasks.put(')]],
  ['workspaces', 'official-host', [route('/api/workspaces', 'GET', '{ workspaces }'),
    route('/api/workspaces', 'POST', 'saveWorkspaces()'), read('/api/workspace', 'workspaceListing(')]],
  ['files', 'official-host', [read('/api/workspace/file', 'text.slice(0, 1024)'),
    route('/api/files/preview', 'GET', 'previewFile('), route('/api/files/raw', 'GET', 'rawFile('),
    route('/api/chat/upload', 'POST', 'uploads.accept(')]],
  ['image-generation', 'official-host', [route('/api/image-models', 'GET', 'imageModels.library()'),
    route('/api/image-models/apply', 'POST', 'imageModels.apply()')]],
  ['updates', 'official-app-control', [route('/api/updates/check', 'GET', 'checkComponentUpdates(')]],
];

export function reviewedSurface(source, appVersion = '0.2.1') {
  if (!REVIEWED_APP_VERSIONS.includes(appVersion))
    throw Object.assign(new Error('official_app_version_needs_validation'), { code: 'official_app_version_needs_validation' });
  const text = compact(source);
  for (const [, , routes] of reviewed) {
    for (const { anchors } of routes) {
      if (!anchors.every(anchor => text.includes(compact(anchor))))
        throw Object.assign(new Error('official_host_contract_changed'), { code: 'official_host_contract_changed' });
    }
  }
  // Construct fresh allowlisted records: no caller metadata, URLs or commands.
  return { schema: 1, appVersion, validation: 'source-anchors-only',
    areas: reviewed.map(([area, owner, routes]) => ({ area, owner, routes: routes.map(({ api }) => ({ ...api })) })) };
}
