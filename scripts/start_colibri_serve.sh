#!/usr/bin/env bash
# Start colibri qwen36 serve (persistent, VRAM retained across requests).
# Usage: ./start_colibri_serve.sh [CUDA_EXPERT_GB] [PORT]
# Defaults: 5  8888
set -e
EXPERT_GB="${1:-5}"
PORT="${2:-8888}"
cd ~/src/colibri
nohup setsid bash -c "nix-shell -p cudaPackages_12.cuda_cudart cudaPackages_12.cuda_cccl python3 gmp --run 'cd c && COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB=$EXPERT_GB HEAT_FILE=/tmp/q36_serve.heat COLI_MODEL=$HOME/models/qwen36-35b-a3b-colibri-i4 python3 ./coli serve --model $HOME/models/qwen36-35b-a3b-colibri-i4 --cap 256 --port $PORT'" > "/tmp/q36_serve_${PORT}.log" 2>&1 &
disown
echo "started colibri serve on port $PORT (log /tmp/q36_serve_${PORT}.log)"
echo "wait ~60s for warmstart, then: ss -tlnp | grep $PORT"
