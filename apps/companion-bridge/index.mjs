#!/usr/bin/env node
import { createBridge, HostClient } from './bridge.mjs';

const options = {};
const args = process.argv.slice(2);
const names = { '--data-dir': 'directory', '--port': 'port', '--admin-port': 'adminPort', '--listen': 'listen', '--advertise-host': 'advertiseHost', '--upstream': 'upstream' };
for (let index = 0; index < args.length; index++) {
  if (args[index] === '--help') {
    console.log('Strata iPhone Companion\n\nStart: node index.mjs [--data-dir PATH] [--port 8787] [--admin-port 8788]\n                       [--advertise-host LAN_IP] [--upstream http://127.0.0.1:PORT]\n\nThe bridge is opt-in. Its HTTPS listener requires pairing. The pairing page is available only on this Mac.\nQuit with Control-C. The existing Strata app and its models are left running.');
    process.exit(0);
  }
  const name = names[args[index]];
  if (!name || !args[index + 1]) {
    console.error('Unknown or incomplete argument. Use --help.'); process.exit(1);
  }
  const value = args[++index];
  if (name === 'port' || name === 'adminPort') {
    const number = Number(value);
    if (!Number.isInteger(number) || number < 1 || number > 65535) { console.error('Choose a port from 1 to 65535.'); process.exit(1); }
    options[name] = number;
  } else options[name] = value;
}
let bridge;
try {
  const { upstream, ...settings } = options;
  bridge = await createBridge({ ...settings, host: new HostClient({ upstream }) });
  // This line is consumed by the native Mac launcher. Do not log invitations or bearer credentials.
  console.log(JSON.stringify({ type: 'ready', adminURL: bridge.adminURL, port: bridge.port, adminPort: bridge.adminPort, fingerprint: bridge.identity.fingerprint }));
} catch (error) {
  console.error(error.code === 'EADDRINUSE' ? 'Strata Companion is already running, or its port is in use.' : error.message);
  process.exit(1);
}
let stopping = false;
async function stop() {
  if (stopping) return;
  stopping = true;
  const timer = setTimeout(() => process.exit(1), 5_000);
  timer.unref();
  await bridge.close(); process.exit(0);
}
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
