#!/bin/bash
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
mkdir -p logs

while true; do
    echo "$(date '+%F %T') starting server on port $PORT" | tee -a logs/server.log
    .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT" 2>&1 | tee -a logs/server.log
    echo "$(date '+%F %T') server stopped, restarting in 3 seconds" | tee -a logs/server.log
    sleep 3
done
