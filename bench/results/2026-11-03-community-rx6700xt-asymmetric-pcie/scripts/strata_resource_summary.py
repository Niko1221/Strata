"""Summarize measured Linux samples; never infer unobserved process memory."""
import json, pathlib, sys

def summarize(rows):
    if not rows:
        return {'samples': 0}
    first, last = rows[0], rows[-1]
    elapsed = last['monotonic'] - first['monotonic']
    result = {'samples': len(rows), 'sample_span_s': elapsed,
              'scope': 'whole machine unless explicitly process_*; background I/O included'}
    result['ram_used_peak_bytes'] = max((r['memory_kib']['MemTotal']-r['memory_kib']['MemAvailable'])*1024 for r in rows)
    swaps = [(r['memory_kib']['SwapTotal']-r['memory_kib']['SwapFree'])*1024 for r in rows]
    result.update(swap_initial_bytes=swaps[0], swap_peak_bytes=max(swaps), swap_change_bytes=swaps[-1]-swaps[0])
    result['gpu'] = {}
    for card in sorted({g['card'] for r in rows for g in r['gpu']}):
        gs = [g for r in rows for g in r['gpu'] if g['card']==card]
        result['gpu'][card] = {}
        for field in ['mem_info_vram_used','gpu_busy_percent','mem_busy_percent']:
            vals = [g[field] for g in gs if g.get(field) is not None]
            if vals:
                result['gpu'][card][field] = {'peak': max(vals), 'sample_mean': sum(vals)/len(vals)}
    # guest and guest_nice are already included in user/nice; omit them.
    ticks = [b-a for a,b in zip(first['cpu_ticks'][:8],last['cpu_ticks'][:8])]
    if sum(ticks)>0:
        result['cpu_busy_percent'] = 100*(sum(ticks)-ticks[3]-ticks[4])/sum(ticks)
        result['cpu_iowait_percent'] = 100*ticks[4]/sum(ticks)
    result['disk'] = {}
    for name, end in last['disk_counters'].items():
        start = first['disk_counters'].get(name)
        if start and elapsed>0:
            result['disk'][name] = {k: (end[k]-start[k])/elapsed for k in ['read_bytes','write_bytes','io_ms']}
    result['disk_units'] = 'bytes/s for read_bytes/write_bytes; io_ms per second, per physical device'
    ps = [r['process'] for r in rows if r.get('process',{}).get('rss_bytes') is not None]
    if ps:
        result['process_rss_peak_bytes'] = max(p['rss_bytes'] for p in ps)
        result['process_pid'] = ps[0]['pid']
    else:
        result['process_rss_peak_bytes'] = None
    return result

if __name__ == '__main__':
    rows = [json.loads(s) for s in pathlib.Path(sys.argv[1]).read_text().splitlines() if s.strip()]
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
