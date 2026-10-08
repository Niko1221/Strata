"""Sample system RAM and the selected validation server's process tree."""
import argparse,json,time
import psutil
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
start=time.monotonic();server=None
with open(a.out,'w') as f:
 while True:
  if server is None:
   for proc in psutil.process_iter(['cmdline']):
    cmd=proc.info['cmdline'] or []
    if 'serve.server' in cmd and a.config in cmd: server=proc;break
  if server is None:
   if time.monotonic()-start>60: raise RuntimeError('server not found')
  else:
   if not server.is_running() or server.status()==psutil.STATUS_ZOMBIE:break
   rss=0
   for proc in [server]+server.children(recursive=True):
    try:rss+=proc.memory_info().rss
    except psutil.NoSuchProcess:pass
   mem=psutil.virtual_memory()
   f.write(json.dumps(dict(elapsed_s=time.monotonic()-start,wall_time=time.time(),available_bytes=mem.available,system_used_bytes=mem.total-mem.available,process_tree_rss_bytes=rss,swap_used_bytes=psutil.swap_memory().used,disk=psutil.disk_io_counters()._asdict()))+'\n');f.flush()
  time.sleep(1)
