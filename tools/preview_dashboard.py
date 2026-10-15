"""Serve the real dashboard with synthetic documentation data, loopback only; no GPU or user files."""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.history import HistoryStore  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.usage import UsageLedger, empty_counts  # noqa: E402


DEMO_NOW = datetime(2026, 1, 15, 12, tzinfo=timezone.utc).timestamp()


class PreviewLedger(UsageLedger):
    def snapshot(self, live=()):
        data = super().snapshot(live)
        data.update(today='2026-01-15', timezone='UTC (example)', timezone_offset_minutes=0,
                    daily_since_date='2026-01-12')
        data['daily'] = [row for row in data['daily'] if row['date'] in self.state['daily']]
        return data


def preview(folder):
    now = DEMO_NOW
    tok = ByteTokenizer()
    model = 'qwen3.8-flash-next (demo)'
    engine = MockEngine(tok, 'This is a synthetic documentation preview, not model inference.', max_context=204800)
    engine.info = {'version': 'demo', 'expert_slots': 4096, 'expert_cache_mib': 8192, 'kv': 'int8'}
    svc = Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'), model_name=model)
    svc.output_history = HistoryStore(folder / 'demo.history.sqlite3')
    svc.usage = PreviewLedger(folder / 'demo.usage.json', model, wall_clock=lambda: now)
    rates = {**svc.usage.state['rates'], 'electricity_per_kwh': 0.15}
    svc.usage.state.update(rates=rates, since=now - 3 * 86400, energy_since=now - 3 * 86400,
                           daily_since=now - 3 * 86400, seeded=True)
    for i in range(3):
        day = (datetime.fromtimestamp(now, timezone.utc) - timedelta(days=i)).date().isoformat()
        counts = {**empty_counts(), 'requests': 42 + i * 7, 'input_tokens': 4_000_000 + i * 500_000,
                  'cached_tokens': 3_600_000 + i * 400_000, 'output_tokens': 140_000 + i * 20_000,
                  'decode_tokens': 140_000 + i * 20_000, 'decode_s': (140_000 + i * 20_000) / 35.2,
                  'active_s': 4200, 'energy_wh': 129.97, 'measured_s': 4200,
                  'cpu_s': 7900, 'cpu_measured_s': 4200, 'energy_method': 'energy counter'}
        svc.usage.state['daily'][day] = counts
        svc.usage._add(svc.usage.state, counts)
    for i, app in enumerate(('Claude Code', 'Trae', 'Strata Web')):
        started = now - 120 - i * 300
        capture = svc.output_history.begin({'id': f'{i + 1:012x}', 'path': '/v1/messages' if i == 0 else '/v1/chat/completions',
            'model': model, 'requested_model': 'claude-opus-4-6' if i == 0 else model,
            'started_at': started, 'client_app': app, 'client_source': 'X-Strata-App',
            'prompt_tokens': 8000 + i * 2000, 'output_tokens': 420 + i * 50, 'stream': True})
        capture.append('reasoning', 'I will preserve the existing layout, check the affected component, and verify the change.\n'
                       'This is synthetic example text for the documentation screenshot.\n')
        capture.append('output', 'Updated the date input component.\n\n'
                       '- Added consistent date formatting.\n- Preserved the existing validation rules.\n'
                       '- Checked the focused unit tests.\n\n'
                       'Example data only. No real project files were read or changed.\n')
        capture.append('tools', '\nTool: Read\n{"file_path":"example/date-input.ts"}\n')
        capture.finish({'state': 'completed', 'finished_at': started + 13.2, 'wallclock_s': 13.2,
                        'http_status': 200, 'finish': 'stop'})
        svc.history.append({'time': started, 'prompt_tokens': 8000 + i * 2000, 'reused': 6500 + i * 1800,
                            'output_tokens': 420 + i * 50, 'decode_tok_s': 35.2 - i, 'prompt_ms': 1200,
                            'decode_ms': 12000, 'finish': 'stop', 'hit_rate': 0.94, 'duration_s': 13.2})
    svc.output_history.flush()
    svc.totals.update(since=now - 3600, requests=3, prompt_tokens=30000, reused=24900, output_tokens=1410,
                      prompt_ms=3600, decode_ms=40057)
    hardware = {'gpu_util': 74, 'gpu_mem_used': 11.2 * 2**30, 'gpu_mem_total': 12 * 2**30,
                'gpu_temp': 62, 'gpu_power': 111.4, 'gpu_power_limit': 180, 'gpu_pcie_gen': 4,
                'gpu_pcie_gen_max': 4, 'gpu_pcie_width': 16, 'gpu_pcie_rx_mb': 4280,
                'cpu': 38, 'ram_used': 54 * 2**30, 'ram_total': 96 * 2**30,
                'disk_read_mb': 12.4, 'disk_write_mb': 0.5}
    history = {key: [max(0, value * (0.85 + 0.15 * math.sin(j / 4))) for j in range(60)]
               for key, value in hardware.items()}
    history.update(tok_s=[34 + 2 * math.sin(j / 5) for j in range(60)],
                   prefill_tok_s_mean=[1250 + 50 * math.sin(j / 5) for j in range(60)])
    svc.telemetry = SimpleNamespace(snapshot=lambda: {'now': hardware, 'history': history,
        'static': {'gpu_name': 'Example NVIDIA GPU (synthetic)', 'cpu_name': 'Example 16-core CPU',
                   'cores': 16, 'threads': 32, 'psutil': True}})
    return svc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8096)
    parser.add_argument('--seconds', type=int, default=300, help='Stop and remove temporary example data after this time')
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error('--seconds must be positive')
    with tempfile.TemporaryDirectory(prefix='strata-doc-preview-') as folder:
        svc = preview(Path(folder))
        httpd = serve(svc, host='127.0.0.1', port=args.port)
        print(f'Synthetic preview: http://127.0.0.1:{httpd.server_address[1]}/#monitor', flush=True)
        print('No native model, credentials, real history or Tailscale connection is used.', flush=True)
        try:
            threading.Event().wait(args.seconds)
        except KeyboardInterrupt:
            pass
        finally:
            httpd.shutdown()
            httpd.server_close()
            svc.output_history.close()
            svc.usage.close()


if __name__ == '__main__':
    main()
