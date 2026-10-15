"""Opt-in, local output history. Disk writes never run on the generation thread."""
from __future__ import annotations

import contextlib
import json
import queue
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

FIELDS = ('output', 'reasoning', 'tools')
META = ('id', 'path', 'model', 'started_at', 'finished_at', 'state', 'stream', 'http_status',
        'usage', 'timings', 'error', 'queue_s', 'load_s', 'first_token_s', 'wallclock_s',
        'finish', 'prompt_tokens', 'output_tokens', 'max_tokens', 'thinking', 'sampling',
        'reasoning_recoveries', 'reasoning_repeat_coverage', 'client_app', 'client_source', 'requested_model')
TERMINAL = ('completed', 'error', 'disconnected', 'interrupted')


def client_identity(headers):
    """A self-reported display label, never an authentication or ownership check."""
    if headers is not None:
        for name in ('X-Strata-App', 'X-Client-Name', 'X-App-Name'):
            value = headers.get(name, '')
            if value and not any(ord(c) < 32 or ord(c) == 127 for c in value):
                label = value.strip()[:80]
                if label:
                    return {'client_app': label, 'client_source': name}
        agent = headers.get('User-Agent', '')[:2048]
        for product, label in (
                ('trae', 'Trae'), ('zcode|z-code', 'Z Code'), ('cursor', 'Cursor'),
                ('windsurf', 'Windsurf'), ('zed', 'Zed'), ('cline', 'Cline'),
                ('roo-code|roo_code', 'Roo Code'), ('opencode', 'OpenCode'),
                ('claude-code|claude-cli', 'Claude Code'),
                ('codex_cli_rs|codex_vscode|codex', 'Codex')):
            if re.search(r'(?:^|[\s(])(?:' + product + r')(?=[/\s;)]|$)', agent, re.IGNORECASE):
                return {'client_app': label, 'client_source': 'User-Agent'}
    return {'client_app': 'Unknown', 'client_source': 'unidentified'}


def safe_meta(record):
    meta = {k: record[k] for k in META if k in record}
    if record.get('_actual_model'):
        meta['requested_model'] = record.get('model')
        meta['model'] = record['_actual_model']
    return meta


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{12}', value):
        raise ValueError('invalid history ID')
    return value


class HistoryStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.error, self.closed = '', False
        self.jobs = queue.Queue(maxsize=256)
        self.active, self.failures = {}, {}
        with contextlib.closing(self._connect()) as db, db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, started REAL, state TEXT, meta TEXT);
                CREATE INDEX IF NOT EXISTS entries_time ON entries(started DESC, id DESC);
                CREATE TABLE IF NOT EXISTS chunks (seq INTEGER PRIMARY KEY, entry_id TEXT, kind TEXT, text TEXT);
                CREATE INDEX IF NOT EXISTS chunks_entry ON chunks(entry_id, kind, seq);
            ''')
            for ident, raw in db.execute("SELECT id,meta FROM entries WHERE state NOT IN (?,?,?,?)", TERMINAL).fetchall():
                meta = json.loads(raw)
                meta['state'] = 'interrupted'
                db.execute('UPDATE entries SET state=?,meta=? WHERE id=?', ('interrupted', json.dumps(meta), ident))
        self.worker = threading.Thread(target=self._work, name='strata-history-writer', daemon=True)
        self.worker.start()

    def _connect(self):
        return sqlite3.connect(str(self.path), timeout=2)

    def _failure(self, ident, error):
        message = f'History recording incomplete for {ident}: {error}'
        if self.error != message:
            print('[strata] ' + message, flush=True)
        self.error = message
        self.failures[ident] = message
        while len(self.failures) > 100:
            self.failures.pop(next(iter(self.failures)))

    def _submit(self, meta, chunks):
        if self.closed:
            return False
        try:
            self.jobs.put_nowait((json.dumps(meta, ensure_ascii=False), chunks))
            return True
        except queue.Full:
            self._failure(meta['id'], 'disk writer queue is full')
            return False

    def _write(self, db, raw, chunks):
        meta = json.loads(raw)
        if meta['id'] in self.failures:
            meta['recording_error'] = self.failures[meta['id']]
            raw = json.dumps(meta, ensure_ascii=False)
        with db:
            db.execute('INSERT INTO entries VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,meta=excluded.meta',
                       (meta['id'], meta['started_at'], meta.get('state', 'queued'), raw))
            db.executemany('INSERT INTO chunks(entry_id,kind,text) VALUES (?,?,?)',
                           ((meta['id'], kind, text) for kind, text in chunks))

    def _work(self):
        db = None
        try:
            while True:
                job = self.jobs.get()
                try:
                    if job is None:
                        return
                    if db is None:
                        db = self._connect()
                    self._write(db, *job)
                except Exception as error:  # noqa: BLE001 - disk failures must not break model requests
                    self._failure(json.loads(job[0])['id'], error)
                finally:
                    self.jobs.task_done()
        finally:
            if db is not None:
                db.close()

    def begin(self, meta):
        capture = HistoryCapture(self, meta)
        self.active[capture.meta['id']] = capture
        capture.flush()
        return capture

    def flush(self):
        for capture in list(self.active.values()):
            capture.flush()
        self.jobs.join()

    def close(self):
        if self.closed:
            return
        for capture in list(self.active.values()):
            capture.flush()
        self.closed = True
        try:
            self.jobs.put(None, timeout=5)
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                self.error = 'History writer did not finish before shutdown'
        except queue.Full:
            self.error = 'History writer did not finish before shutdown'

    def _meta(self, raw):
        meta = json.loads(raw)
        meta.setdefault('client_app', 'Unknown')
        meta.setdefault('client_source', 'not_recorded')
        if meta['id'] in self.failures:
            meta['recording_error'] = self.failures[meta['id']]
        return meta

    def records(self, before=None, query='', state='all'):
        where, values = [], []
        with contextlib.closing(self._connect()) as db:
            if before:
                valid_id(before)
                row = db.execute('SELECT started FROM entries WHERE id=?', (before,)).fetchone()
                if row is None:
                    raise ValueError('history page cursor no longer exists')
                where.append('(started < ? OR (started = ? AND id < ?))')
                values.extend((row[0], row[0], before))
            if state != 'all':
                if state not in (*TERMINAL, 'active'):
                    raise ValueError('invalid history status')
                if state == 'active':
                    where.append('state NOT IN (?,?,?,?)')
                    values.extend(TERMINAL)
                else:
                    where.append('state=?')
                    values.append(state)
            if query:
                needle = query[:200].casefold()

                class TextMatch:
                    def __init__(self):
                        self.tail, self.found = '', False

                    def step(self, text):
                        if not self.found:
                            combined = self.tail + text.casefold()
                            self.found = needle in combined
                            self.tail = combined[-(len(needle) - 1):] if len(needle) > 1 else ''

                    def finalize(self):
                        return self.found

                db.create_aggregate('history_match', 1, TextMatch)
                pattern = '%' + query[:200].replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
                where.append("(meta LIKE ? ESCAPE '\\' OR id IN (SELECT entry_id FROM "
                             "(SELECT entry_id,kind,text FROM chunks ORDER BY entry_id,kind,seq) "
                             "GROUP BY entry_id,kind HAVING history_match(text)))")
                values.append(pattern)
            clause = ' WHERE ' + ' AND '.join(where) if where else ''
            rows = db.execute('SELECT meta FROM entries' + clause + ' ORDER BY started DESC,id DESC LIMIT 51', values).fetchall()
        records = [self._meta(row[0]) for row in rows[:50]]
        return {'requests': records, 'next_cursor': records[-1]['id'] if len(rows) > 50 else None,
                'persistent': True, 'error': self.error or None}

    def detail(self, ident, field='output', offset=0):
        valid_id(ident)
        if field not in FIELDS or offset < 0:
            raise ValueError('invalid history field or offset')
        with contextlib.closing(self._connect()) as db:
            row = db.execute('SELECT meta FROM entries WHERE id=?', (ident,)).fetchone()
            if row is None:
                return None
            pieces, count, skip = [], 0, offset
            for (text,) in db.execute('SELECT text FROM chunks WHERE entry_id=? AND kind=? ORDER BY seq', (ident, field)):
                if skip >= len(text):
                    skip -= len(text)
                    continue
                piece, skip = text[skip:skip + 65537 - count], 0
                pieces.append(piece)
                count += len(piece)
                if count > 65536:
                    break
        text = ''.join(pieces)
        return {'entry': self._meta(row[0]), 'field': field, 'text': text[:65536],
                'next_offset': offset + 65536 if len(text) > 65536 else None}

    def export(self, ident):
        valid_id(ident)
        with contextlib.closing(self._connect()) as db:
            row = db.execute('SELECT meta FROM entries WHERE id=?', (ident,)).fetchone()
            if row is None:
                return
            yield 'STRATA OUTPUT HISTORY\n\n' + json.dumps(self._meta(row[0]), ensure_ascii=False, indent=2) + '\n'
            for kind in ('reasoning', 'output', 'tools'):
                yield '\n=== ' + kind.upper() + ' ===\n\n'
                for (text,) in db.execute('SELECT text FROM chunks WHERE entry_id=? AND kind=? ORDER BY seq', (ident, kind)):
                    yield text
                yield '\n'


class HistoryCapture:
    def __init__(self, store, meta):
        self.store, self.meta = store, safe_meta(meta)
        self.meta.setdefault('id', uuid.uuid4().hex[:12])
        self.meta.setdefault('started_at', time.time())
        valid_id(self.meta['id'])
        self.chunks, self.size, self.at, self.failed = [], 0, time.monotonic(), False

    def append(self, kind, text):
        if self.failed or not text or kind not in FIELDS:
            return
        for start in range(0, len(text), 16384):
            piece = text[start:start + 16384]
            self.chunks.append((kind, piece))
            self.size += len(piece)
            if self.size >= 16384 or time.monotonic() - self.at >= 1:
                self.flush()

    def flush(self, meta=None):
        if meta:
            self.meta.update(safe_meta(meta))
        if not self.failed:
            self.failed = not self.store._submit(dict(self.meta), self.chunks)
        self.chunks, self.size, self.at = [], 0, time.monotonic()

    def finish(self, meta):
        self.flush(meta)
        self.store.active.pop(self.meta['id'], None)


def event_parts(api, item):
    """Output already being sent to the client; never request headers or credentials."""
    output, reasoning, tools, meta = '', '', '', {}
    if api == 'openai':
        choice = (item.get('choices') or [{}])[0]
        delta = choice.get('delta', {})
        output, reasoning = delta.get('content', ''), delta.get('reasoning_content', '')
        for call in delta.get('tool_calls', []):
            fn = call.get('function', {})
            if fn.get('name'):
                tools += '\nTool: ' + fn['name'] + '\n'
            tools += fn.get('arguments', '')
        if item.get('strata_mcp'):
            tools += '\nMCP: ' + json.dumps(item['strata_mcp'], ensure_ascii=False) + '\n'
        meta = {k: item[k] for k in ('usage', 'timings') if item.get(k)}
        if choice.get('finish_reason'):
            meta['finish'] = choice['finish_reason']
    elif api == 'responses':
        kind = item.get('type')
        output = item.get('delta', '') if kind == 'response.output_text.delta' else ''
        reasoning = item.get('delta', '') if kind == 'response.reasoning_text.delta' else ''
        if kind == 'response.output_item.added' and item.get('item', {}).get('type') == 'function_call':
            tools = '\nTool: ' + item['item'].get('name', '') + '\n'
        elif kind == 'response.function_call_arguments.delta':
            tools = item.get('delta', '')
        if item.get('response', {}).get('usage'):
            meta['usage'] = item['response']['usage']
    else:
        _, event = item
        delta, block = event.get('delta', {}), event.get('content_block', {})
        output, reasoning = delta.get('text', ''), delta.get('thinking', '')
        if block.get('type') == 'tool_use':
            tools = '\nTool: ' + block.get('name', '') + '\n'
            if block.get('input'):
                tools += json.dumps(block['input'], ensure_ascii=False)
        tools += delta.get('partial_json', '')
        if event.get('usage'):
            meta['usage'] = event['usage']
        if delta.get('stop_reason'):
            meta['finish'] = delta['stop_reason']
    return {'output': output or '', 'reasoning': reasoning or '', 'tools': tools or ''}, meta
