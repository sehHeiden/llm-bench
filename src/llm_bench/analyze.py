"""
Parse results/bench.log into Markdown comparison tables.

Usage: uv run analyze [--log results/bench.log] [--out results/table.md]
"""

import argparse
import re
from pathlib import Path


def _seg(domain: str, backend: str, txt: str) -> str:
    """Return the log segment belonging to one series."""
    start = re.search(rf"\[{domain} \| {backend}\]", txt)
    if not start:
        return ""
    seg = txt[start.end() :]
    nxt = re.search(r"\[(\w+) \| ", seg)
    return seg[: nxt.start()] if nxt else seg


def _g(seg: str, pat: str) -> str:
    m = re.search(pat, seg)
    return m.group(1) if m else "-"


def _series(domain: str, backend: str, txt: str) -> dict[str, str]:
    s = _seg(domain, backend, txt)
    return {
        "domain": domain,
        "backend": backend,
        "cold_tps": _g(s, r"(?m)^cold: ([\d.]+) tok"),
        "warm_tps": _g(s, r"warm (?:tok/s )?mean: ([\d.]+)"),
        "warm_sd": _g(s, r"warm (?:tok/s )?mean stddev: ([\d.]+)"),
        "cold_vhit": _g(s, r"cold: [\d.]+ tok/s TTFT [\d.]+s VRAMhit ([\d.]+)%"),
        "warm_vhit": _g(s, r"warm VRAMhit % mean: ([\d.]+)"),
        "warm_vhit_sd": _g(s, r"warm VRAMhit % mean stddev: ([\d.]+)"),
        "cpu_warm": _g(s, r"CPU mean: ([\d.]+)%"),
        "gpu_warm": _g(s, r"GPU mean: ([\d.]+)%"),
        "vram_warm": _g(s, r"VRAM mean: ([\d.]+) GB"),
    }


def _table(rows: list[dict[str, str]]) -> str:
    header = (
        "| Domäne | Backend | cold tok/s | warm tok/s ± | cold VRAMhit | warm VRAMhit ± | "
        "CPU warm | GPU warm | VRAM warm |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for r in rows:
        if r["backend"] == "llama":
            lines.append(
                f"| {r['domain']} | llama | - | {r['warm_tps']} ({r['warm_sd']}) | - | - | "
                f"{[r['cpu_warm']]} | {[r['gpu_warm']]} | {[r['vram_warm']]} |"
            )
        else:
            lines.append(
                f"| {r['domain']} | {r['backend']} | {r['cold_tps']} | {r['warm_tps']} ({r['warm_sd']}) | "
                f"{r['cold_vhit']} | {r['warm_vhit']} ({r['warm_vhit_sd']}) | "
                f"{r['cpu_warm']} | {r['gpu_warm']} | {r['vram_warm']} |"
            )
    return "\n".join(lines)


def _build_tables(txt: str, domains: list[str], backends: list[str]) -> str:
    rows = []
    for domain in domains:
        for backend in backends:
            row = _series(domain, backend, txt)
            if row["warm_tps"] != "-" or row["cold_tps"] != "-":
                rows.append(row)
    return _table(rows)


def main() -> None:
    """Parse bench.log and write comparison table."""
    ap = argparse.ArgumentParser(description="Analyze bench.log into a Markdown table")
    ap.add_argument("--log", default="results/bench.log")
    ap.add_argument("--out", default="results/table.md")
    args = ap.parse_args()

    txt = Path(args.log).read_text()
    domains = ["math", "geography", "history", "philosophy", "physics", "chemistry"]
    backends = ["llama", "colibri-int4", "colibri-gs64"]
    table = _build_tables(txt, domains, backends)

    out = Path(args.out)
    out.write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
