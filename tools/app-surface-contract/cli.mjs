#!/usr/bin/env node
/** Explicit offline commands only. No auto-detection, daemon or activation. */
import {parseArgs} from 'node:util';
import {inspectApp, readCapabilities} from './contract.mjs';
import {exportCandidate, verifyCandidate} from './assets.mjs';

try {
  const {values, positionals} = parseArgs({allowPositionals:true, options:{
    host:{type:'string'}, output:{type:'string'}, candidate:{type:'string'},
    'manifest-sha256':{type:'string'}, help:{type:'boolean'}}});
  if (values.help) {
    process.stdout.write('App Surface Contract\n\n'
      + 'snapshot --host <Installed Host Directory>\n'
      + 'export --host <Installed Host Directory> --output <New Candidate Directory>\n'
      + 'verify --host <Installed Host Directory> --candidate <Candidate Directory> --manifest-sha256 <Pinned SHA256>\n');
  } else {
    const [command, ...extra] = positionals;
    const permitted = {snapshot:['host'], export:['host','output'], verify:['host','candidate','manifest-sha256']}[command];
    if (!permitted || extra.length || permitted.some(key => !values[key])
        || Object.keys(values).some(key => !permitted.includes(key))) throw Error('invalid_arguments');
    const inspected = await inspectApp(values.host);
    const snapshot = await readCapabilities(inspected);
    const result = command === 'snapshot' ? snapshot : command === 'export'
      ? await exportCandidate(inspected, values.output, snapshot)
      : await verifyCandidate(values.candidate, {manifestSha256:values['manifest-sha256'], current:snapshot});
    process.stdout.write(JSON.stringify(result) + '\n');
  }
} catch (error) {
  process.stdout.write(JSON.stringify({schema:1, state:'needs_validation', reason:
    /^[a-z_]+$/.test(error.message) ? error.message : 'app_contract_unavailable'}) + '\n');
  process.exitCode = 1;
}
