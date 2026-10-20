import { json, error, readBody } from './util.mjs';
import { isLocalCompanionRequest } from './companion-routes.mjs';

export function createConversationRoutes({ conversations, port }) {
  return async (req, res, url) => {
    res.removeHeader('access-control-allow-origin');
    res.setHeader('cache-control', 'no-store');
    if (!isLocalCompanionRequest(req, typeof port === 'function' ? port() : port)) return error(res, 403, 'Open Strata on this Mac to use conversations.');
    const root = '/api/companion/conversations';
    try {
      if (url.pathname === root && req.method === 'GET') return json(res, 200, conversations.list());
      if (url.pathname === root && req.method === 'POST') return json(res, 201, conversations.create(await readBody(req)));
      if (url.pathname === root + '/sync' && req.method === 'POST') return json(res, 200, conversations.sync((await readBody(req, 64 * 1024 * 1024)).sessions));
      const match = /^\/api\/companion\/conversations\/([A-Za-z0-9_.:-]{1,160})(?:\/(messages|stop|approval|preferences))?$/.exec(url.pathname);
      if (!match) return error(res, 404, 'Unknown conversation action.');
      const [, id, action] = match;
      if (!action && req.method === 'GET') return json(res, 200, conversations.detail(id));
      if (!action && req.method === 'DELETE') return json(res, 200, conversations.remove(id));
      if (action === 'preferences' && req.method === 'PATCH') return json(res, 200, conversations.preferences(id, await readBody(req)));
      if (action === 'messages' && req.method === 'POST') return json(res, 202, conversations.send(id, (await readBody(req)).message));
      if (action === 'stop' && req.method === 'POST') return json(res, 200, await conversations.stop(id));
      if (action === 'approval' && req.method === 'POST') { const body = await readBody(req); return json(res, 200, await conversations.approval(id, body.toolId, body.decision)); }
      return error(res, 405, 'This conversation action is unavailable.');
    } catch (e) { return error(res, e.status ?? 400, e.message ?? 'The conversation could not be updated.'); }
  };
}


/** Shared transcripts are private even though legacy operational events allow other local clients. */
export function addConversationEvents({ req, res, hub, conversations, port }) {
  const privateEligible = isLocalCompanionRequest(req, typeof port === 'function' ? port() : port);
  hub.add(res, { allowEvent: event => event !== 'conversation' || privateEligible });
  if (!privateEligible) return;
  // Initial snapshots close the gap between bootstrap and the live connection, under the same access check.
  for (const conversation of conversations.list().conversations) {
    res.write(`event: conversation\ndata: ${JSON.stringify({ conversation })}\n\n`);
  }
}
