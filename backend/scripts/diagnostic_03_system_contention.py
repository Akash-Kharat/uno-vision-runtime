import subprocess
import sys
import time
import json
import threading
from pathlib import Path
try:
    import psutil
except ImportError:
    psutil = None

def get_cpu_frequencies():
    freqs = []
    try:
        if psutil:
            for freq in psutil.cpu_freq(percpu=True):
                freqs.append(freq.current)
    except Exception:
        pass
    
    # Fallback to sysfs if psutil fails or missing
    if not freqs:
        for i in range(16): # Check up to 16 cores
            path = Path(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq")
            if path.exists():
                try:
                    val = int(path.read_text().strip()) / 1000.0 # MHz
                    freqs.append(val)
                except Exception:
                    pass
    return freqs

def get_thermal_info():
    zones = {}
    try:
        thermal_dir = Path("/sys/class/thermal")
        if thermal_dir.exists():
            for zone in thermal_dir.glob("thermal_zone*"):
                try:
                    temp = int((zone / "temp").read_text().strip()) / 1000.0
                    type_name = (zone / "type").read_text().strip()
                    zones[type_name] = temp
                except Exception:
                    pass
    except Exception:
        pass
    return zones

def get_system_load():
    try:
        with open("/proc/loadavg", "r") as f:
            return f.read().strip()
    except Exception:
        return "N/A"

stats_history = []
stop_monitoring = False

def monitor_loop(pid: int):
    global stop_monitoring
    process = None
    if psutil:
        try:
            process = psutil.Process(pid)
        except Exception:
            pass
            
    while not stop_monitoring:
        stat = {
            "time": time.time(),
            "loadavg": get_system_load(),
            "freqs_mhz": get_cpu_frequencies(),
            "thermals_c": get_thermal_info(),
        }
        if psutil:
            stat["cpu_percent"] = psutil.cpu_percent(interval=None, percpu=True)
            if process:
                try:
                    stat["proc_cpu_percent"] = process.cpu_percent(interval=None)
                    stat["proc_threads"] = process.num_threads()
                    stat["proc_rss_mb"] = process.memory_info().rss / (1024*1024)
                except Exception:
                    pass
        stats_history.append(stat)
        time.sleep(1.0)

def main():
    print("==================================================")
    print("DIAGNOSTIC 03: SYSTEM CONTENTION MONITORING")
    print("==================================================")
    
    print("\nPre-benchmark baseline (5 seconds):")
    for _ in range(5):
        print(f"Load: {get_system_load()} | Freqs: {get_cpu_frequencies()} | Thermals: {get_thermal_info()}")
        time.sleep(1)
        
    print("\nStarting production benchmark...")
    benchmark_script = Path(__file__).resolve().parent / "benchmark_detection.py"
    
    # Start benchmark
    cmd = [sys.executable, str(benchmark_script), "--warmup", "10", "--iterations", "30"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    global stop_monitoring
    monitor_thread = threading.Thread(target=monitor_loop, args=(proc.pid,))
    monitor_thread.start()
    
    # Stream output
    for line in proc.stdout:
        print(line, end="")
        
    proc.wait()
    stop_monitoring = True
    monitor_thread.join()
    
    print("\nPost-benchmark baseline (5 seconds):")
    for _ in range(5):
        print(f"Load: {get_system_load()} | Freqs: {get_cpu_frequencies()} | Thermals: {get_thermal_info()}")
        time.sleep(1)
        
    # Summarize stats
    print("\n--- Summary of During-Benchmark Stats ---")
    if stats_history:
        try:
            avg_proc_cpu = sum(s.get("proc_cpu_percent", 0) for s in stats_history) / len(stats_history)
            max_rss = max(s.get("proc_rss_mb", 0) for s in stats_history)
            avg_threads = sum(s.get("proc_threads", 0) for s in stats_history) / len(stats_history)
            print(f"Avg Process CPU %:  {avg_proc_cpu:.2f}%")
            print(f"Max Process RSS:    {max_rss:.2f} MB")
            print(f"Avg Process Threads: {avg_threads:.1f}")
        except Exception:
            print("Process detailed stats (psutil) not available.")
            
        print("\nThermal Peak:")
        for k in stats_history[0].get("thermals_c", {}).keys():
            peak = max(s.get("thermals_c", {}).get(k, 0) for s in stats_history)
            print(f"  {k}: {peak:.2f} C")
            
        print("\nFrequency Range:")
        if stats_history[0].get("freqs_mhz"):
            cores = len(stats_history[0]["freqs_mhz"])
            for c in range(cores):
                freqs = [s["freqs_mhz"][c] for s in stats_history if len(s["freqs_mhz"]) > c]
                print(f"  Core {c}: Min {min(freqs):.1f} MHz, Max {max(freqs):.1f} MHz")

if __name__ == "__main__":
    main()
