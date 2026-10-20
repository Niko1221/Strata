import { json, error } from './util.mjs';

// Pairing invitations grant control of this Mac. Unlike the general host API,
// these routes accept only this exact desktop origin and loopback socket peers.
export function isLocalCompanionRequest(req, port) {
  const peer = req.socket?.remoteAddress ?? '';
  if (!/^(127\.\d+\.\d+\.\d+|::1|::ffff:127\.\d+\.\d+\.\d+)$/.test(peer)) return false;
  try {
    const local = new URL(`http://${req.headers.host}`);
    if (!['127.0.0.1', 'localhost', '[::1]'].includes(local.hostname) || local.port !== String(port)
        || local.username || local.password || local.pathname !== '/' || local.search || local.hash) return false;
    const origin = req.headers.origin;
    if (origin !== undefined && (typeof origin !== 'string' || origin !== local.origin)) return false;
    return true;
  } catch { return false; }
}

export function createCompanionRoutes({ manager, port }) {
  return async function companionRoutes(req, res, url) {
    res.removeHeader('access-control-allow-origin');
    res.setHeader('cache-control', 'no-store');
    res.setHeader('x-content-type-options', 'nosniff');
    res.setHeader('referrer-policy', 'no-referrer');
    if (!isLocalCompanionRequest(req, typeof port === 'function' ? port() : port)) {
      return error(res, 403, 'Open Pair iPhone in the Strata app on this Mac.');
    }
    const path = url.pathname;
    const method = req.method;
    try {
      if (path === '/api/companion' && method === 'GET') return json(res, 200, manager.state());
      if (path === '/api/companion/start' && method === 'POST') return json(res, 200, await manager.start());
      if (path === '/api/companion/stop' && method === 'POST') return json(res, 200, await manager.stop());
      if (path === '/api/companion/invitation' && method === 'POST') return json(res, 200, await manager.rotateInvitation());
      if (path === '/api/companion/qr' && method === 'GET') {
        const png = manager.qrImage();
        if (!png) return error(res, 404, 'Create a new pairing invitation first.');
        res.writeHead(200, { 'content-type': 'image/png', 'content-length': png.length });
        res.end(png); return;
      }
      const device = /^\/api\/companion\/devices\/([a-zA-Z0-9_-]{1,100})$/.exec(path);
      if (device && method === 'DELETE') return json(res, 200, await manager.revokeSession(device[1]));
      return error(res, 404, 'Unknown iPhone pairing action.');
    } catch (failure) {
      return error(res, failure.status ?? 503, failure.status ? failure.message : 'iPhone pairing is unavailable. Try again.');
    }
  };
}
