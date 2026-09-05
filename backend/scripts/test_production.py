import time
import requests
import psutil
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=1800)
    args = parser.parse_args()

    # Start camera
    r = requests.post("http://127.0.0.1:8000/api/v1/camera/start")
    print("Camera start:", r.json())
    
    # Start inference
    r = requests.post("http://127.0.0.1:8000/api/v1/runtime/start")
    print("Inference start:", r.json())
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < args.duration:
            time.sleep(5)
            
            # Fetch status
            r = requests.get("http://127.0.0.1:8000/api/v1/runtime/status")
            if r.status_code == 200:
                data = r.json()
                stats = data.get("runtime", {}).get("stats", {})
                
                inf_fps = stats.get("effective_fps", 0)
                captured = stats.get("frames_captured", 0)
                inferred = stats.get("frames_inferred", 0)
                skipped = stats.get("frames_skipped", 0)
                
                cpu = psutil.cpu_percent()
                try:
                    temps = psutil.sensors_temperatures()
                    temp = list(temps.values())[0][0].current if temps else 0
                except:
                    temp = 0
                    
                age = stats.get("latest_detection_age_ms")
                age_str = f"{age:.1f}ms" if age is not None else "N/A"
                
                print(f"Elapsed: {int(time.time() - start_time)}s | FPS: {inf_fps:.2f} | "
                      f"Capt: {captured}, Inf: {inferred}, Skip: {skipped} | "
                      f"Age: {age_str} | CPU: {cpu}% | Temp: {temp}C")
            else:
                print("Failed to get status:", r.status_code)
                
    except KeyboardInterrupt:
        pass
        
    requests.post("http://127.0.0.1:8000/api/v1/runtime/stop")
    requests.post("http://127.0.0.1:8000/api/v1/camera/stop")
    
if __name__ == "__main__":
    main()
