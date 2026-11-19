/** Deterministic comparison of supplied CSV data. No file, shell or network access. */
function csv(text, required) {
  if (typeof text !== 'string' || Buffer.byteLength(text) > 1024 * 1024) throw new Error('Supply a complete CSV of at most 1 MiB.');
  const rows = []; let row = [], cell = '', quoted = false, closed = false;
  const value = () => { row.push(cell.trim()); cell = ''; closed = false; if (row.length > 64) throw new Error('Too many CSV columns.'); };
  const line = () => { value(); if (row.some(Boolean)) rows.push(row); row = []; if (rows.length > 10001) throw new Error('Too many CSV rows.'); };
  text = text.replace(/^\uFEFF/, '');
  // The read tool renders every physical line as N: content. Accept only a
  // complete, consecutive display beginning at line 1; partial views fail closed.
  if (/^1: /.test(text)) {
    const lines = text.split(/\r?\n/);
    if (lines.at(-1) === '') lines.pop();
    if (!lines.every((line, i) => line.startsWith((i + 1) + ': ')))
      throw new Error('Supply the complete CSV without display line numbers.');
    text = lines.map((line, i) => line.slice(String(i + 1).length + 2)).join('\n');
  }
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) { if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (ch === '"') { quoted = false; closed = true; } else cell += ch; continue; }
    if (ch === ',') value();
    else if (ch === '\n' || ch === '\r') { if (ch === '\r' && text[i + 1] === '\n') i++; line(); }
    else if (ch === '"' && cell === '' && !closed) quoted = true;
    else { if (closed || ch === '"') throw new Error('Malformed CSV quoting.'); cell += ch; }
  }
  if (quoted) throw new Error('Unclosed CSV quote.');
  if (cell || row.length || closed) line();
  const header = rows.shift();
  if (!header || new Set(header).size !== header.length || required.some(key => !header.includes(key))) throw new Error(`CSV requires unique headers: ${required.join(', ')}.`);
  return rows.map(values => {
    if (values.length !== header.length) throw new Error('CSV row does not match its header.');
    const result = Object.fromEntries(header.map((key, i) => [key, values[i]]));
    if (required.some(key => !result[key])) throw new Error('CSV required values cannot be empty.');
    return result;
  });
}

export function auditInventory({ inventory_csv, expected_csv, requirements_csv }) {
  const tables = {
    inventory: csv(inventory_csv, ['id', 'version']),
    expected: csv(expected_csv, ['id', 'expected_version']),
    requirements: csv(requirements_csv, ['id', 'required_check', 'status']),
  };
  const grouped = Object.fromEntries(Object.entries(tables).map(([name, rows]) => {
    const map = new Map(); for (const row of rows) { if (!map.has(row.id)) map.set(row.id, []); map.get(row.id).push(row); } return [name, map];
  }));
  const duplicates = [];
  for (const [table, rows] of Object.entries(tables)) {
    const seen = new Map();
    for (const row of rows) { const key = table === 'requirements' ? JSON.stringify([row.id, row.required_check]) : row.id; seen.set(key, (seen.get(key) ?? 0) + 1); }
    for (const [key, count] of seen) if (count > 1) duplicates.push({ table, id: table === 'requirements' ? JSON.parse(key)[0] : key, count, ...(table === 'requirements' ? { required_check: JSON.parse(key)[1] } : {}) });
  }
  const ids = [...new Set(Object.values(tables).flatMap(rows => rows.map(row => row.id)))].sort();
  const missing_ids = [], version_mismatches = [];
  for (const id of ids) {
    const missingFrom = Object.keys(tables).filter(table => !grouped[table].has(id));
    if (missingFrom.length) missing_ids.push({ id, missingFrom });
    const actual = grouped.inventory.get(id), expected = grouped.expected.get(id);
    // Ambiguous duplicate IDs are reported as duplicates, never arbitrarily matched.
    if (actual?.length === 1 && expected?.length === 1 && actual[0].version !== expected[0].expected_version)
      version_mismatches.push({ id, inventory_version: actual[0].version, expected_version: expected[0].expected_version });
  }
  const failed_checks = tables.requirements.filter(row => row.status.toLowerCase() !== 'passed');
  return { criteria: 'Exact version strings by ID; passed status is successful without additional evidence. Requirements may have multiple distinct checks per ID.',
    version_mismatches, duplicates, missing_ids, failed_checks,
    total_mismatches: version_mismatches.length, total_duplicates: duplicates.length, total_missing: missing_ids.length, total_failed: failed_checks.length };
}
