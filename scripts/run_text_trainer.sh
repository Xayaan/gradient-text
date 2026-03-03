#!/bin/bash
set -e
cd /workspace/scripts
# Winner-compatible mode: start redis when available unless file backend is forced.
STATE_BACKEND="${TEXT_MINER_STATE_BACKEND:-auto}"
if [ "${STATE_BACKEND}" != "file" ] && command -v redis-server >/dev/null 2>&1; then
  redis-server --daemonize yes || true
  sleep 2
fi
python3 /workspace/scripts/text_trainer.py "$@"
