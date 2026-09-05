#!/bin/bash
# Launch uvicorn in background, wait for it, then run the validation.
cd ~/uno-vision-runtime/backend

# Kill any stale uvicorn
pkill -f uvicorn 2>/dev/null || true
sleep 2

# Start the API
nohup venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/uvicorn.log 2>&1 &
echo "Uvicorn PID: $!"
sleep 8  # wait for startup

# Run the 30-minute validation (1800 seconds)
venv/bin/python3 scripts/board_validate.py 1800 2>&1 | tee /tmp/validation.log
echo "Validation complete."
