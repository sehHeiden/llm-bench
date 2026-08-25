"""
Orchestrator: run all benchmark series (6 domains x 3 backends).

Protocol per domain: Q1=cold, Q2-20=heat buildup (discarded), Q21-30=warm (mean+stddev).

Modes:
  - llama:         persistent serve (Q1-30 series, no cold/hot)
  - colibri int4:  standalone (fresh process per prompt, heat via HEAT_FILE) — VRAM hit rate
  - colibri gs64:  standalone (same) — VRAM hit rate

Usage: uv run run-all [--port 8888] [--domains math,geography,...]
"""

import argparse
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOMAINS_ALL = ["math", "geography", "history", "philosophy", "physics", "chemistry"]
LLAMA_LOAD = 45


def _kill_server() -> None:
    """Kill any running server processes and wait for port to free."""
    subprocess.run(["pkill", "-9", "-f", "qwen36"], check=False)
    subprocess.run(["pkill", "-9", "-f", "coli serve"], check=False)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
    time.sleep(5)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=REPO)


def _series(domain: str, backend: str, port: int, log: Path) -> None:
    """Run one benchmark series and append to log."""
    header = f"[{domain} | {backend}]"
    print(f">>> {header} ({time.strftime('%H:%M:%S')})")
    with log.open("a") as fh:
        fh.write(f"\n{header}\n")

    if backend == "llama":
        _kill_server()
        _run(["uv", "run", "bench", "start", "--backend", "llama", "--port", str(port)])
        time.sleep(LLAMA_LOAD)
        cmd = [
            "uv",
            "run",
            "bench",
            "serve",
            "--prompts",
            f"prompts/{domain}.txt",
            "--mode",
            "llama",
            "--model",
            "qwen3.6-35b-a3b",
        ]
        r = _run(cmd)
    else:  # colibri standalone — fresh process per prompt, heat accumulates over Q1-Q30
        container = backend.removeprefix("colibri-")
        snap = (
            Path.home()
            / "models"
            / ("qwen36-35b-a3b-colibri-i4-gs64" if container == "gs64" else "qwen36-35b-a3b-colibri-i4")
        )
        heat_file = f"/tmp/llm_bench_{backend}.heat"
        inner = (
            f"export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1; "
            f"cd {REPO} && uv run bench standalone --prompts prompts/{domain}.txt "
            f"--snap {snap} --heat {heat_file}"
        )
        nix_shell = (
            "nix-shell -p cudaPackages_12.cuda_cudart cudaPackages_12.cuda_cccl "
            f"cudaPackages_12.libcublas gmp --run '{inner}'"
        )
        r = _run(["bash", "-lc", nix_shell])
    with log.open("a") as fh:
        fh.write(r.stdout + r.stderr + "\n")
    # free GPU/CPU/VRAM for the next series — a lingering llama server would
    # contend for resources and corrupt the next series' CPU/GPU/VRAM samples
    _kill_server()


def main() -> None:
    """Run benchmark series (6 domains x selected backends)."""
    ap = argparse.ArgumentParser(description="Run benchmark series")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--domains", default=",".join(DOMAINS_ALL))
    ap.add_argument(
        "--backends",
        default="llama,colibri-int4,colibri-gs64",
        help="comma-separated: llama / colibri-int4 / colibri-gs64",
    )
    args = ap.parse_args()

    domains = args.domains.split(",")
    backends = args.backends.split(",")
    log = REPO / "results" / "bench.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"=== llm-bench results ({time.strftime('%Y-%m-%dT%H:%M:%S')}) ===\n"
        f"Protocol: Q1-10 EN, Q11-20 DE, Q21-30 FR. "
        f"colibri: Q1=cold, Q2-20=heat, Q21-30=warm (standalone, VRAM hit rate). "
        f"llama: Q1-30=series (serve).\n"
        f"GTX 1660 Ti / 6GB / sm_75, CUDA_EXPERT_GB=5\n"
        f"Backends: {','.join(backends)}\n\n"
    )
    log.write_text(header)

    for domain in domains:
        for backend in backends:
            _series(domain, backend, args.port, log)

    _kill_server()
    with log.open("a") as fh:
        fh.write(f"=== done {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
    print(f">>> ALL DONE — see {log}")


if __name__ == "__main__":
    main()
