"""Start board_validate.py detached on the board, then poll /tmp/validation.log."""
import paramiko, time, sys

HOST = "192.168.31.216"
USER = "arduino"
PASS = "Micro@4545"

def ssh_detach(cmd, wait=2):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=10)
    transport = client.get_transport()
    channel = transport.open_session()
    channel.exec_command(cmd)
    time.sleep(wait)
    channel.close()
    client.close()

def ssh_run(cmd, timeout=10):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=10)
    _, stdout, stderr = client.exec_command(cmd)
    stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()
    client.close()
    return out, err

# Launch validation detached
print("Launching 30-min validation detached...")
validate_cmd = (
    "setsid nohup bash -c '"
    "cd ~/uno-vision-runtime/backend && "
    "venv/bin/python3 scripts/board_validate.py 1800 "
    "> /tmp/validation.log 2>&1"
    "' &"
)
ssh_detach(validate_cmd, wait=3)
print("Validation started. Polling every 60s...")

# Poll the log
duration = 1800 + 120  # 30 min + buffer
poll_start = time.time()
last_lines = 0

while time.time() - poll_start < duration:
    time.sleep(60)
    elapsed = int(time.time() - poll_start)
    out, _ = ssh_run("tail -20 /tmp/validation.log 2>&1")
    lines = out.strip().splitlines()
    print(f"\n--- [{elapsed}s] /tmp/validation.log (last 20 lines) ---")
    for line in lines:
        print(line)

    # Stop early if "Validation complete" appears
    if "DONE" in out or "Validation complete" in out:
        print("\n==> Validation completed!")
        break

print("\nFetching full results summary...")
out, _ = ssh_run("cat /tmp/validation.log")
print(out[-5000:] if len(out) > 5000 else out)
