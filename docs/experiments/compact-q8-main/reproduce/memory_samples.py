from pathlib import Path
import ctypes,json,re,threading,time
class Memory(ctypes.Structure):
    _fields_=[('total',ctypes.c_uint64),('free',ctypes.c_uint64),('used',ctypes.c_uint64)]
def start(out,engine_getter):
    nv=ctypes.CDLL('libnvidia-ml.so.1');assert nv.nvmlInit_v2()==0
    handle=ctypes.c_void_p();assert nv.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(handle))==0
    nv.nvmlDeviceGetMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory)]
    stop=threading.Event();rows=[]
    def work():
        while not stop.is_set():
            m=Memory();error=nv.nvmlDeviceGetMemoryInfo(handle,ctypes.byref(m))
            info={k:int(v)*1024 for k,v in re.findall(r'^(\w+):\s+(\d+) kB',Path('/proc/meminfo').read_text(),re.M)}
            row=dict(unix=time.time(),gpu_error=error,gpu_free_bytes=m.free,gpu_used_bytes=m.used,
                host_available_bytes=info['MemAvailable'],host_swap_used_bytes=info['SwapTotal']-info['SwapFree'])
            engine=engine_getter()
            if engine is not None:
                try:
                    status=Path(f'/proc/{engine.proc.pid}/status').read_text()
                    row.update({k:int(v)*1024 for k,v in re.findall(r'^(VmRSS|VmSwap):\s+(\d+) kB',status,re.M)})
                except FileNotFoundError:pass
            rows.append(row);stop.wait(.2)
    thread=threading.Thread(target=work,daemon=True);thread.start()
    def finish():
        stop.set();thread.join(timeout=5);nv.nvmlShutdown()
        good=[x for x in rows if not x['gpu_error']];assert good
        summary=dict(sampled_gpu_min_free_bytes=min(x['gpu_free_bytes'] for x in good),
            sampled_host_min_available_bytes=min(x['host_available_bytes'] for x in rows),
            sampled_host_max_swap_used_bytes=max(x['host_swap_used_bytes'] for x in rows),
            sampled_engine_max_swap_bytes=max(x.get('VmSwap',0) for x in rows),
            scope='200ms samples across startup,function,benchmark,shutdown; not allocator-exact peaks. NVML GPU memory plus proc host/engine memory.')
        (out/'memory-samples.json').write_text(json.dumps(dict(summary=summary,samples=rows),indent=2)+'\n')
    return finish
