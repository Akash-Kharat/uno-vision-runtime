import requests
import time

def run():
    print("Gathering timing info from API...")
    for _ in range(5):
        r = requests.get("http://localhost:8000/api/v1/runtime/status")
        stats = r.json().get("runtime", {}).get("stats", {})
        print(f"avg_cap: {stats.get('avg_capture_ms')}, avg_pre: {stats.get('avg_preprocessing_ms')}, avg_inf: {stats.get('avg_inference_ms')}, avg_post: {stats.get('avg_postprocessing_ms')}, avg_tot: {stats.get('avg_total_ms')}")
        time.sleep(1)

if __name__ == '__main__':
    run()
