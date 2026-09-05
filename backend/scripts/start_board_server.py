"""Start uvicorn + validation on the board, detached from SSH session."""
import paramiko, time, sys

HOST = "192.168.31.216"
USER = "arduino"
PASS = "Micro@4545"

def ssh_detach(cmd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=10)
    transport = client.get_transport()
    channel = transport.open_session()
    # Use nohup with setsid so the process lives after channel close
    channel.exec_command(cmd)
    time.sleep(3)
    channel.close()
    client.close()

# Step 1: Kill any stale uvicorn
print("Killing stale uvicorn...")
ssh_detach("pkill -f uvicorn 2>/dev/null; sleep 1")
time.sleep(3)

# Step 2: Start uvicorn detached
print("Starting uvicorn...")
uvicorn_cmd = (
    "setsid nohup "
    "bash -c 'cd ~/uno-vision-runtime/backend && "
    "venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 "
    "> /tmp/uvicorn.log 2>&1' "
    "&"
)
ssh_detach(uvicorn_cmd)
print("Uvicorn launch sent. Waiting 15s for startup...")
time.sleep(15)

# Step 3: Check it's up
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)
_, stdout, _ = client.exec_command("curl -s http://127.0.0.1:8000/ 2>&1")
out = stdout.read().decode().strip()
client.close()
if "UNO Vision" in out or "api_version" in out:
    print("API is UP:", out)
else:
    print("API check:", out or "(no response yet)")
    
print("\nBoard uvicorn started. Now start the validation:")
print("  Run: python run_remote.py \"cd ~/uno-vision-runtime/backend && venv/bin/python3 scripts/board_validate.py 1800 > /tmp/validation.log 2>&1\"")
