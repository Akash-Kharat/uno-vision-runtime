"""Upload board_validate.py and board_launch.sh, then start the 30-min validation via SSH (non-blocking)."""
import paramiko
import sys
import time

HOST = "192.168.31.216"
USER = "arduino"
PASS = "Micro@4545"

files_to_upload = [
    ("scripts/board_validate.py", "/home/arduino/uno-vision-runtime/backend/scripts/board_validate.py"),
    ("scripts/board_launch.sh",   "/home/arduino/uno-vision-runtime/backend/scripts/board_launch.sh"),
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)
sftp = client.open_sftp()

for local, remote in files_to_upload:
    sftp.put(local, remote)
    print(f"Uploaded {local} -> {remote}")

sftp.close()

# Run: chmod script, kill stale uvicorn, start uvicorn in background, wait, then start validation in background too
cmd = """
pkill -f uvicorn 2>/dev/null || true
sleep 2
chmod +x ~/uno-vision-runtime/backend/scripts/board_launch.sh
nohup bash ~/uno-vision-runtime/backend/scripts/board_launch.sh > /tmp/board_launch_out.log 2>&1 &
echo "Launch PID: $!"
"""
stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
time.sleep(5)
out = stdout.read().decode()
err = stderr.read().decode()
print("STDOUT:", out)
print("STDERR:", err)
client.close()
print("Board launch initiated. Check /tmp/board_launch_out.log on the board for progress.")
