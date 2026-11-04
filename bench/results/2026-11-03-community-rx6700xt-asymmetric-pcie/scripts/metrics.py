"""Read-only Linux system samples. Counters are cumulative; rates require deltas."""
import argparse,json,pathlib,time,os
P=pathlib.Path
def sample(pid=None):
 out={'time_unix':time.time(),'monotonic':time.monotonic()}
 out['memory_kib']={k:int(v.split()[0]) for k,v in (l.split(':',1) for l in P('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable','SwapTotal','SwapFree','Cached','Dirty']}
 out['cpu_ticks']=[int(x) for x in P('/proc/stat').read_text().splitlines()[0].split()[1:]]
 out['clock_ticks_per_second']=os.sysconf('SC_CLK_TCK')
 out['gpu']=[]
 for d in sorted(P('/sys/class/drm').glob('card[0-9]*/device')):
  if not (d/'mem_info_vram_total').exists():continue
  g={'card':d.parent.name}
  for key in ['mem_info_vram_used','mem_info_vram_total','gpu_busy_percent','mem_busy_percent']:
   try:g[key]=int((d/key).read_text())
   except (OSError,ValueError):g[key]=None
  out['gpu'].append(g)
 out['disk_counters']={}
 for line in P('/proc/diskstats').read_text().splitlines():
  x=line.split();name=x[2]
  if not name.startswith(('nvme','sd')) or (P('/sys/class/block')/name/'partition').exists():continue
  out['disk_counters'][name]={'read_ios':int(x[3]),'read_bytes':int(x[5])*512,'write_ios':int(x[7]),'write_bytes':int(x[9])*512,'io_ms':int(x[12])}
 if pid:
  try:
   base=P('/proc')/str(pid);st=base.joinpath('stat').read_text().rsplit(')',1)[1].split()
   out['process']={'pid':pid,'user_ticks':int(st[11]),'system_ticks':int(st[12]),'rss_bytes':int(st[21])*os.sysconf('SC_PAGE_SIZE')}
   out['process']['io']={k:int(v) for k,v in (l.split(':') for l in base.joinpath('io').read_text().splitlines())}
  except (OSError,ValueError):out['process']={'pid':pid,'available':False}
 return out
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--pid',type=int);a.add_argument('--seconds',type=int,default=3);a.add_argument('--out',required=True);args=a.parse_args()
 with open(args.out,'x') as f:
  for _ in range(args.seconds):f.write(json.dumps(sample(args.pid))+'\n');f.flush();time.sleep(1)
