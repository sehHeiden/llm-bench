"""
Benchmark a persistent OpenAI-compatible serve endpoint (colibri or llama.cpp).

Usage: uv run serve-bench --url URL --model MODEL --prompts FILE [--max-tokens N]
Defaults: url=http://127.0.0.1:8888/v1/chat/completions  model=glm-5.2-colibri
          prompts=prompts/math.txt  max-tokens=100
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def main() -> None:
    """Run the serve benchmark against a persistent OpenAI-compatible endpoint."""
    ap = argparse.ArgumentParser(description="Persistent serve benchmark")
    ap.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
    ap.add_argument("--model", default="glm-5.2-colibri")
    ap.add_argument("--prompts", default="prompts/math.txt")
    ap.add_argument("--max-tokens", type=int, default=100)
    args = ap.parse_args()

    prompts = Path(args.prompts).read_text().strip().splitlines()
    print("req tok/s tokens wall_s")
    for i, p in enumerate(prompts):
        body = json.dumps(
            {
                "model": args.model,
                "messages": [{"role": "user", "content": p}],
                "max_tokens": args.max_tokens,
                "stream": False,
            }
        ).encode()
        t0 = time.time()
        r = urllib.request.urlopen(
            urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"}),
            timeout=180,
        )
        raw = r.read()
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as e:
            sys.exit(f"[req {i}] bad JSON: {e} · raw[:200]={raw[:200]!r}")
        dt = time.time() - t0
        gen = d.get("usage", {}).get("completion_tokens", 0)
        tps = gen / dt if dt > 0 else 0
        print(f"{i} {tps:.2f} {gen} {dt:.2f}")


if __name__ == "__main__":
    main()
