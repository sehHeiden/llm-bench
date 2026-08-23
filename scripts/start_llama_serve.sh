#!/usr/bin/env bash
# Start llama.cpp serve (--cpu-moe, persistent).
# Usage: ./start_llama_serve.sh [PORT]
# Defaults: 8888
set -e
PORT="${1:-8888}"
nohup ~/bin/llama-server-qwen.sh > "/tmp/llama_serve_${PORT}.log" 2>&1 &
disown
echo "started llama serve on port $PORT (log /tmp/llama_serve_${PORT}.log)"
echo "wait ~45s for model load, then: ss -tlnp | grep $PORT"
