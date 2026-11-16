/** Ordinary stdio MCP fixture: three enumerated CSV reads and one fixed audit output.
 * No model-controlled paths, commands, URLs, or dependencies. Not a sandbox qualification.
 */
import {createHash} from 'node:crypto';
import {closeSync, constants, fstatSync, lstatSync, openSync, readFileSync, realpathSync, writeFileSync} from 'node:fs';
import {dirname, isAbsolute, join, parse, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

export const INPUTS = Object.freeze(['inventory', 'expected', 'requirements']);
export const TOOLS = Object.freeze([
  {name: 'read_audit_csv', description: 'Read one complete synthetic CSV for audit_inventory. Only the three named inputs exist.',
    inputSchema: {type: 'object', properties: {table: {type: 'string', enum: INPUTS}}, required: ['table'], additionalProperties: false},
    annotations: {readOnlyHint: true}},
  {name: 'save_audit', description: 'Persist the complete audit_inventory JSON result to the fixed audit.json output. No path argument; exactly one successful save.',
    inputSchema: {type: 'object', properties: {audit_json: {type: 'string', maxLength: 65536}}, required: ['audit_json'], additionalProperties: false},
    annotations: {readOnlyHint: false, destructiveHint: false}},
]);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const fail = () => { throw new Error('Bounded audit fixture rejected the request'); };

// Recheck every ancestor and final pathname on every request. The owned directory
// is created by the controller; model tools cannot rename it or create links.
export function noLinks(path) {
  if (!isAbsolute(path)) fail();
  let current = resolve(path);
  for (;;) {
    const stat = lstatSync(current);
    if (stat.isSymbolicLink()) fail();
    if (current === parse(current).root) break;
    current = dirname(current);
  }
  if (resolve(realpathSync(path)).toLowerCase() !== resolve(path).toLowerCase()) fail();
}

function exactArgs(args, key) {
  if (!args || Array.isArray(args) || typeof args !== 'object' || Object.keys(args).length !== 1 || !Object.hasOwn(args, key)) fail();
}

export function createFixture(root) {
  root = resolve(root);
  noLinks(root);
  if (!lstatSync(root).isDirectory()) fail();
  const read = table => {
    if (!INPUTS.includes(table)) fail();
    noLinks(root);
    const path = join(root, table + '.csv');
    noLinks(path);
    const fd = openSync(path, constants.O_RDONLY);
    try {
      const stat = fstatSync(fd);
      if (!stat.isFile() || stat.size > 65536 || stat.nlink !== 1) fail();
      return readFileSync(fd);
    } finally { closeSync(fd); }
  };
  const initial = Object.fromEntries(INPUTS.map(table => [table, hash(read(table))]));
  let saved = false;
  return function call(name, args) {
    if (name === 'read_audit_csv') {
      exactArgs(args, 'table');
      const bytes = read(args.table);
      if (hash(bytes) !== initial[args.table]) fail();
      return {content: [{type: 'text', text: bytes.toString('utf8')}]};
    }
    if (name !== 'save_audit') fail();
    exactArgs(args, 'audit_json');
    if (saved || typeof args.audit_json !== 'string' || Buffer.byteLength(args.audit_json) > 65536) fail();
    const value = JSON.parse(args.audit_json);
    if (!value || Array.isArray(value) || typeof value !== 'object') fail();
    const pairs = [['version_mismatches', 'total_mismatches'], ['duplicates', 'total_duplicates'], ['missing_ids', 'total_missing'], ['failed_checks', 'total_failed']];
    if (typeof value.criteria !== 'string' || pairs.some(([rows, total]) => !Array.isArray(value[rows]) || !Number.isSafeInteger(value[total]) || value[total] !== value[rows].length)) fail();
    for (const table of INPUTS) if (hash(read(table)) !== initial[table]) fail();
    noLinks(root);
    // Exclusive creation rejects symlink/hardlink/pre-existing output. Never overwrite.
    const bytes = Buffer.from(JSON.stringify(value, null, 2) + '\n');
    writeFileSync(join(root, 'audit.json'), bytes, {flag: 'wx'});
    saved = true;
    return {content: [{type: 'text', text: JSON.stringify({saved: 'audit.json', bytes: bytes.length, sha256: hash(bytes)})}]};
  };
}

export async function serve(root, input = process.stdin, output = process.stdout) {
  const call = createFixture(root);
  const send = message => output.write(JSON.stringify({jsonrpc: '2.0', ...message}) + '\n');
  let pending = '';
  // Bounded input parsing avoids an unbounded readline buffer. No raw request or
  // exception text is printed: arguments and filesystem error paths stay private.
  for await (const chunk of input) {
    pending += chunk.toString('utf8');
    if (Buffer.byteLength(pending) > 262144) fail();
    let end;
    while ((end = pending.indexOf('\n')) !== -1) {
      const line = pending.slice(0, end); pending = pending.slice(end + 1);
      let msg;
      try { msg = JSON.parse(line); } catch { send({id: null, error: {code: -32700, message: 'Invalid JSON'}}); continue; }
      if (msg.id === undefined) continue;
      if (!['string', 'number'].includes(typeof msg.id)) { send({id: null, error: {code: -32600, message: 'Invalid request'}}); continue; }
      try {
        let result;
        if (msg.method === 'initialize') result = {protocolVersion: msg.params?.protocolVersion ?? '2024-11-05', capabilities: {tools: {listChanged: false}}, serverInfo: {name: 'strata-owned-audit', version: '1.0.0'}};
        else if (msg.method === 'ping') result = {};
        else if (msg.method === 'tools/list') result = {tools: TOOLS};
        else if (msg.method === 'tools/call') {
          try { result = call(msg.params?.name, msg.params?.arguments); }
          catch { result = {isError: true, content: [{type: 'text', text: 'Bounded audit fixture rejected the request'}]}; }
        } else { send({id: msg.id, error: {code: -32601, message: 'Method unavailable'}}); continue; }
        send({id: msg.id, result});
      } catch { send({id: msg.id, error: {code: -32603, message: 'Fixture request failed'}}); }
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.length !== 3) process.exitCode = 2;
  else await serve(process.argv[2]).catch(() => { process.stderr.write('Bounded audit fixture stopped\n'); process.exitCode = 1; });
}
