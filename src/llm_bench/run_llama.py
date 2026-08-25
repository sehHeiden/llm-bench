"""
Run the llama.cpp baseline: start server, wait for readiness, bench, stop, push.

Usage: uv run run-llama [--prompts prompts/math.txt] [--port 8888]
"""

import argparse
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "qwen3.6-35b-a3b"


def main() -> None:
    """Start llama, bench the prompt set, stop, push result to ntfy."""
    ap = argparse.ArgumentParser(description="llama.cpp baseline benchmark")
    ap.add_argument("--prompts", default="prompts/math.txt")
    ap.add_argument("--port", type=int, default=8888)
    args = ap.parse_args()

    subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
    time.sleep(2)

    subprocess.run(
        ["uv", "run", "bench", "start", "--backend", "llama", "--port", str(args.port)],
        check=False,
        cwd=REPO,
    )
    time.sleep(45)  # model load

    r = subprocess.run(
        [
            "uv",
            "run",
            "bench",
            "serve",
            "--mode",
            "llama",
            "--model",
            MODEL,
            "--prompts",
            args.prompts,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    out = Path(REPO) / "results" / "llama_baseline.txt"
    out.write_text(r.stdout + r.stderr)

    subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)

    # extract warm mean + push
    for line in r.stdout.splitlines():
        if "warm mean" in line:
            from .notify import notify

            notify(f"llama baseline: {line.strip()}")
            break
    print(f">>> llama done — see {out}")


if __name__ == "__main__":
    main()
