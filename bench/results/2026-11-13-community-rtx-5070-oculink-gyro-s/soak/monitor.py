"""Samples the graphics card's own sensors (power, temperature, load, memory, fan), CPU load, free RAM and whether the
server port answers, every INTERVAL seconds, into power-log.csv. Stops when a file named monitor.stop appears.
Power is the RTX 5070's board power as reported by nvidia-smi: it does not include the CPU, the eGPU dock's losses
or the rest of the PC."""
import csv, ctypes, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, free_ram_gb, port_open

INTERVAL = 5
OUT = os.path.join(HERE, "power-log.csv"); STOP = os.path.join(HERE, "monitor.stop")
Q = "power.draw,temperature.gpu,utilization.gpu,memory.used,fan.speed"

class FT(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]
def cpu_times():
    i, k, u = FT(), FT(), FT(); ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
    v = lambda f: (f.hi << 32) | f.lo
    return v(i), v(k) + v(u)          # idle, total (kernel time includes idle)

def gpu():
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20, creationflags=0x08000000).stdout.strip().splitlines()[0]
        return [x.strip() for x in out.split(",")]
    except Exception:
        return [""] * 5

new = not os.path.exists(OUT)
with open(OUT, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new: w.writerow(["epoch", "time", "gpu_power_w", "gpu_temp_c", "gpu_util_pct", "vram_used_mib", "fan_pct", "cpu_pct", "ram_free_gb", "port_8091"])
    idle0, tot0 = cpu_times()
    while not os.path.exists(STOP):
        t = time.time(); g = gpu(); idle1, tot1 = cpu_times()
        cpu = round(100 * (1 - (idle1 - idle0) / max(1, tot1 - tot0)), 1); idle0, tot0 = idle1, tot1
        w.writerow([round(t, 1), time.strftime("%H:%M:%S", time.localtime(t))] + g + [cpu, free_ram_gb(), int(port_open())]); f.flush()
        time.sleep(max(0.5, INTERVAL - (time.time() - t)))
