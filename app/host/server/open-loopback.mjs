/**
 * Open a loopback HTTP(S) URL in the person's default browser.
 * Only 127.0.0.1 / [::1] with an explicit validated port are accepted.
 * Never puts credentials in the URL or command line.
 */
import { spawn } from 'node:child_process';
import { externalEnvironment } from './external-env.mjs';

const LOOPBACK = /^(https?):\/\/(127\.0\.0\.1|\[::1\])(?::(\d+))?(\/[\w./?&=-]*)?$/i;

export function parseLoopbackUrl(raw) {
  if (typeof raw !== 'string' || !raw.trim()) throw Object.assign(new Error('Choose a local address.'), { code: 'bad-url' });
  const match = LOOPBACK.exec(raw.trim());
  if (!match) throw Object.assign(new Error('Only a local 127.0.0.1 address can be opened this way.'), { code: 'bad-url' });
  const protocol = match[1].toLowerCase();
  const host = match[2];
  const port = match[3] ? Number(match[3]) : (protocol === 'https' ? 443 : 80);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw Object.assign(new Error('Choose a local port from 1024 to 65535.'), { code: 'bad-port' });
  }
  const path = match[4] || '/';
  const href = `${protocol}://${host}:${port}${path}`;
  return { protocol, host, port, path, href };
}

export function openLoopbackCommand(url, platform = process.platform) {
  const { href } = typeof url === 'string' ? parseLoopbackUrl(url) : url;
  if (platform === 'win32') return ['rundll32.exe', ['url.dll,FileProtocolHandler', href]];
  if (platform === 'darwin') return ['open', [href]];
  return ['xdg-open', [href]];
}

export function openLoopbackUrl(raw, { platform = process.platform, recorder = null, spawnProcess = spawn } = {}) {
  const parsed = parseLoopbackUrl(raw);
  const [command, args] = openLoopbackCommand(parsed, platform);
  if (process.env.STRATA_OPENER === 'record' || recorder) {
    (recorder ?? openLoopbackUrl.recorded).push([command, ...args]);
    return { ok: true, started: true, url: parsed.href, port: parsed.port };
  }
  const child = spawnProcess(command, args, { detached: true, stdio: 'ignore', windowsHide: false, env: externalEnvironment() });
  child.on('error', () => {});
  child.unref?.();
  return { ok: true, started: true, url: parsed.href, port: parsed.port };
}
openLoopbackUrl.recorded = [];
