#!/usr/bin/env python3
"""Run colibri qwen36 standalone (fresh process per prompt), heat accumulates via HEAT_FILE.
Parses tok/s, TTFT, VRAM-hit, CPU-miss, RAM-hit from stderr per run.

Usage: uv run standalone-bench --prompts FILE [--expert-gb N] [--engine PATH] [--snap PATH]
Defaults: prompts=prompts/math.txt  expert-gb=5
          engine=~/src/colibri/c/qwen36.real  snap=~/models/qwen36-35b-a3b-colibri-i4
"""
import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="colibri standalone benchmark")
    ap.add_argument("--prompts", default="prompts/math.txt")
    ap.add_argument("--expert-gb", default="5")
    ap.add_argument("--engine", default=str(Path.home() / "src/colibri/c/qwen36.real"))
    ap.add_argument("--snap", default=str(Path.home() / "models/qwen36-35b-a3b-colibri-i4"))
    ap.add_argument("--n-new", default="100")
    args = ap.parse_args()

    prompts = Path(args.prompts).read_text().strip().splitlines()
    heat = Path("/tmp/q36_standalone.heat")
    if heat.exists():
        heat.unlink()

    print("run tok/s TTFT_s VRAMhit% CPUmiss RAMhit%")
    for i, p in enumerate(prompts):
        pf = Path(tempfile.mkstemp(suffix=".txt")[1])
        pf.write_text(p + "\n")
        env = {
            **os.environ,
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "SNAP": args.snap, "COLI_CUDA": "1", "COLI_GPUS": "0",
            "CUDA_EXPERT_GB": args.expert_gb, "N_NEW": args.n_new,
            "HEAT_FILE": str(heat),
        }
        r = subprocess.run([args.engine, "256", "4", str(pf)],
                           capture_output=True, text=True, env=env)
        pf.unlink()
        err = r.stderr
        speed = re.search(r"^Speed:\s*([\d.]+)", err, re.M)
        ttft = re.search(r"TTFT:\s*([\d.]+)", err)
        vhit = re.search(r"VRAM hit rate:\s*([\d.]+)", err)
        rhit = re.search(r"Expert cache hit rate:\s*([\d.]+)", err)
        cmiss = re.search(r"miss\(CPU\)\s*(\d+)", err)
        s = speed.group(1) if speed else "-"
        t = ttft.group(1) if ttft else "-"
        v = vhit.group(1) if vhit else "-"
        c = cmiss.group(1) if cmiss else "-"
        h = rhit.group(1) if rhit else "-"
        print(f"{i} {s} {t} {v} {c} {h}")


if __name__ == "__main__":
    main()
