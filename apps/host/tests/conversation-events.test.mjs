import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { EventHub } from '../server/util.mjs';
import { addConversationEvents } from '../server/conversation-routes.mjs';

class Subscriber extends EventEmitter {
  output = '';
  headers = null;
  writeHead(status, headers) { this.headers = { status, ...headers }; }
  write(frame) { this.output += frame; }
}
const request = (host = '127.0.0.1:47831', origin, remoteAddress = '127.0.0.1') => ({ headers: { host, ...(origin === undefined ? {} : { origin }) }, socket: { remoteAddress } });

test('legacy EventHub clients retain all events unless an explicit per-client filter is supplied', t => {
  const hub = new EventHub();
  const legacy = new Subscriber(), filtered = new Subscriber();
  t.after(() => { legacy.emit('close'); filtered.emit('close'); });
  hub.add(legacy);
  hub.add(filtered, { allowEvent: event => event !== 'private' });
  hub.send('private', { text: 'private-value' });
  hub.send('monitor', { healthy: true });
  assert.match(legacy.output, /private-value/);
  assert.doesNotMatch(filtered.output, /private-value/);
  assert.match(legacy.output, /event: monitor/);
  assert.match(filtered.output, /event: monitor/);
  assert.equal(hub.size, 2);
  legacy.emit('close'); assert.equal(hub.size, 1);
});

test('initial and live conversation events require exact desktop origin, loopback Host and loopback peer', t => {
  const hub = new EventHub();
  const conversation = { id: 'private-chat', messages: [{ content: 'INITIAL-PRIVATE-HISTORY' }] };
  const conversations = { list: () => ({ conversations: [conversation] }) };
  const cases = [
    ['same-origin Mac', request('127.0.0.1:47831', 'http://127.0.0.1:47831'), true],
    ['native loopback client', request(), true],
    ['IPv6 Mac', request('[::1]:47831', 'http://[::1]:47831', '::1'), true],
    ['foreign website', request('127.0.0.1:47831', 'https://malicious.example'), false],
    ['other loopback website', request('127.0.0.1:47831', 'http://127.0.0.1:9999'), false],
    ['opaque origin', request('127.0.0.1:47831', 'null'), false],
    ['DNS rebinding', request('rebound.example:47831', 'http://rebound.example:47831'), false],
    ['LAN subscriber', request('127.0.0.1:47831', undefined, '192.168.1.20'), false],
    ['wrong host port', request('127.0.0.1:9999'), false],
  ];
  const subscribers = cases.map(([label, req, allowed]) => {
    const res = new Subscriber(); t.after(() => res.emit('close'));
    addConversationEvents({ req, res, hub, conversations, port: 47831 });
    assert.equal(res.output.includes('INITIAL-PRIVATE-HISTORY'), allowed, `${label}: initial snapshot`);
    return { label, res, allowed };
  });
  hub.send('conversation', { conversation: { ...conversation, messages: [{ content: 'LIVE-PRIVATE-HISTORY' }] } });
  hub.send('monitor', { publicOperationalState: 'healthy' });
  for (const { label, res, allowed } of subscribers) {
    assert.equal(res.output.includes('LIVE-PRIVATE-HISTORY'), allowed, `${label}: live snapshot`);
    assert.ok(res.output.includes('healthy'), `${label}: existing operational events still arrive`);
    if (!allowed) assert.doesNotMatch(res.output, /private-chat|event: conversation/, `${label}: no transcript identifiers or deletion events`);
  }
});
