"""
Benchmark and server launcher for OpenAI-compatible endpoints (serve + standalone).

Protocol per prompts file (30 questions: Q1-10 EN, Q11-20 DE, Q21-30 FR):
  Q1 = cold measurement, Q2-Q20 = heat buildup (discarded), Q21-Q30 = warm measurements
  (mean + stddev). Tracks CPU%/GPU%/VRAM during warm questions via a sampler thread.

Subcommands:
  start       — launch a persistent server (colibri or llama) in background via nix-shell
  serve       — benchmark a running server (colibri or llama), Q1/Q2-20/Q21-30 protocol
  standalone  — fresh process per prompt, heat accumulates via HEAT_FILE (colibri only)
  run-all     — orchestrate all series (6 domains x backends) → results/bench.log
  analyze     — parse results/bench.log → Markdown table (results/table.md)

Usage:
  uv run bench start --backend colibri [--container int4] [--port 8888]
  uv run bench serve --mode llama --model qwen3.6-35b-a3b --prompts prompts/math.txt
  uv run bench standalone --snap ~/models/qwen36-35b-a3b-colibri-i4 --prompts prompts/math.txt
  uv run bench run-all [--domains math,geography,...] [--backends llama,colibri-int4,colibri-gs64]
  uv run bench analyze [--log results/bench.log] [--out results/table.md]
"""

import argparse
import contextlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import psutil
from pydantic import BaseModel, ValidationError

from .notify import notify

N_PROMPTS = 30
HEAT_LAST = 19  # Q2-Q20 (index 1-19) build heat, discarded
STDDEV_MIN = 2
RETRY_MAX = 3
RETRY_WAIT = 10
HTTP_OK = 200
HTTP_SERVICE_UNAVAILABLE = 503

COLIBRI_INT4 = Path.home() / "models/qwen36-35b-a3b-colibri-i4"
COLIBRI_GS64 = Path.home() / "models/qwen36-35b-a3b-colibri-i4-gs64"
LLAMA_WRAPPER = Path.home() / "bin/llama-server-qwen.sh"
REPO = Path(__file__).resolve().parents[2]
DOMAINS = ["math", "geography", "history", "philosophy", "physics", "chemistry"]
BACKENDS = ["llama", "colibri-int4", "colibri-gs64"]
LLAMA_LOAD = 45  # seconds for the llama model to load before series start


class ResourceSampler:
    """Background thread sampling CPU% and GPU%+VRAM during bench."""

    _GPU_NSMI_ARGS = (
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    )
    _GPU_MIN_PARTS = 2

    def __init__(self) -> None:
        """Initialize empty sample lists and stop event."""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu: list[float] = []
        self.gpu: list[float] = []
        self.vram_mb: list[float] = []

    def start(self) -> None:
        """Start the sampler thread."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the sampler thread and join."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        psutil.cpu_percent(interval=None)  # prime
        while not self._stop.is_set():
            self.cpu.append(psutil.cpu_percent(interval=None))
            self._sample_gpu()
            self._stop.wait(1.0)

    def _sample_gpu(self) -> None:
        try:
            r = subprocess.run(self._GPU_NSMI_ARGS, capture_output=True, text=True, check=False, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return
        if r.returncode != 0 or not r.stdout.strip():
            return
        parts = r.stdout.strip().split(", ")
        if len(parts) < self._GPU_MIN_PARTS:
            return
        try:
            self.gpu.append(float(parts[0]))
            self.vram_mb.append(float(parts[1]))
        except ValueError:
            return  # malformed output, skip sample

    @staticmethod
    def mean_std(values: list[float]) -> tuple[float | None, float | None]:
        """Return (mean, stddev) of a sample list, or (None, None) if empty."""
        if not values:
            return None, None
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) >= STDDEV_MIN else 0.0
        return round(mean, 1), round(std, 1)

    def report(self) -> str:
        """Return formatted mean+stddev for CPU, GPU, VRAM."""
        lines = []
        cpu_m, cpu_s = self.mean_std(self.cpu)
        gpu_m, gpu_s = self.mean_std(self.gpu)
        vram_m, vram_s = self.mean_std(self.vram_mb)
        if cpu_m is not None:
            lines.append(f"CPU mean: {cpu_m}% (stddev {cpu_s}%, n={len(self.cpu)})")
        if gpu_m is not None:
            lines.append(f"GPU mean: {gpu_m}% (stddev {gpu_s}%, n={len(self.gpu)})")
        if vram_m is not None and vram_s is not None:
            lines.append(f"VRAM mean: {vram_m / 1024:.2f} GB (stddev {vram_s / 1024:.2f} GB)")
        return "\n".join(lines) if lines else "no samples"


def _grep(pattern: str, text: str, *, multiline: bool = False) -> str:
    """Return first regex group match or '-'."""
    flags = re.MULTILINE if multiline else 0
    m = re.search(pattern, text, flags)
    return m.group(1) if m else "-"


def _mean_std(values: list[float], label: str) -> None:
    """Print mean and stddev of a metric list."""
    if not values:
        print(f"{label}: n/a")
        return
    print(f"{label}: {statistics.mean(values):.2f}")
    if len(values) >= STDDEV_MIN:
        print(f"{label} stddev: {statistics.stdev(values):.2f}")


# ---- serve (persistent server) ----


def _request(url: str, model: str, prompt: str, max_tokens: int, idx: int) -> float:
    """Send one request, return tok/s. Retry on 503/connection errors."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode()
    for attempt in range(RETRY_MAX):
        t0 = time.time()
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}),
                timeout=300,
            )
        except urllib.error.HTTPError as e:
            if e.code != HTTP_SERVICE_UNAVAILABLE:
                raise
            if attempt >= RETRY_MAX - 1:
                raise
            print(f"  [req {idx}] 503, retry {attempt + 1}/{RETRY_MAX} in {RETRY_WAIT}s...")
            time.sleep(RETRY_WAIT)
            continue
        except urllib.error.URLError as e:
            if attempt >= RETRY_MAX - 1:
                raise
            print(f"  [req {idx}] connection error: {e}, retry {attempt + 1}/{RETRY_MAX} in {RETRY_WAIT}s...")
            time.sleep(RETRY_WAIT)
            continue
        raw = r.read()
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as e:
            sys.exit(f"[req {idx}] bad JSON: {e} · raw[:200]={raw[:200]!r}")
        dt = time.time() - t0
        gen = d.get("usage", {}).get("completion_tokens", 0)
        return gen / dt if dt > 0 else 0.0
    return 0.0


def _wait_for_port(port: int, timeout: int = 180) -> bool:
    """Poll until /v1/models returns 200 (server fully ready) or timeout."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == HTTP_OK:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(5)
    return False


def cmd_start(args: argparse.Namespace) -> None:
    """Launch a server (colibri or llama) in background via nix-shell."""
    port = args.port
    if Backend(args.backend) is Backend.LLAMA:
        log = Path(f"/tmp/llama_serve_{port}.log")
        with log.open("w") as fh:
            subprocess.Popen(
                [str(LLAMA_WRAPPER)],
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"started llama serve on port {port} (log {log})")
    else:  # colibri
        snap = str(args.snap or (COLIBRI_GS64 if Container(args.container) is Container.GS64 else COLIBRI_INT4))
        expert_gb = args.expert_gb
        log = Path(f"/tmp/q36_serve_{port}.log")
        cmd = (
            f"nix-shell -p cudaPackages_12.cuda_cudart cudaPackages_12.cuda_cccl "
            f"cudaPackages_12.libcublas python3 gmp --run "
            f"'cd ~/src/colibri/c && COLI_CUDA=1 COLI_GPUS=0 CUDA_EXPERT_GB={expert_gb} "
            f"HEAT_FILE=/tmp/q36_serve.heat COLI_MODEL={snap} "
            f"python3 ./coli serve --model {snap} --cap 256 --port {port}'"
        )
        with log.open("w") as fh:
            subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "HOME": str(Path.home())},
            )
        print(f"started colibri serve ({args.container}) on port {port} (log {log})")
    print(f"waiting for port {port}...")
    if _wait_for_port(port):
        print(f"✓ port {port} listening")
    else:
        print(f"✗ port {port} not up after 180s — check {log}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Benchmark a running server (colibri or llama), Q1/Q2-20/Q21-30 protocol."""
    prompts = Path(args.prompts).read_text().strip().splitlines()
    if len(prompts) < N_PROMPTS:
        msg = f"need {N_PROMPTS} prompts, got {len(prompts)}"
        raise SystemExit(msg)

    cold: float | None = None
    warm: list[float] = []
    sampler = ResourceSampler()
    sampler_started = False
    mode = Backend(args.mode)

    for i, p in enumerate(prompts):
        tps = _request(args.url, args.model, p, args.max_tokens, i)
        if mode is Backend.COLIBRI:
            if i == 0:
                cold = tps
                print(f"cold: req{i} {tps:.2f} tok/s")
            elif 1 <= i <= HEAT_LAST:
                print(f"heat: req{i} {tps:.2f} tok/s (discarded)")
            else:
                if not sampler_started:
                    sampler.start()
                    sampler_started = True
                warm.append(tps)
                print(f"warm: req{i} {tps:.2f} tok/s")
        else:
            warm.append(tps)
            print(f"req{i} {tps:.2f} tok/s")

    sampler.stop()
    print("---")
    if cold is not None:
        print(f"cold: {cold:.2f} tok/s")
    warm_mean = statistics.mean(warm) if warm else None
    warm_std = statistics.stdev(warm) if len(warm) >= STDDEV_MIN else None
    if warm_mean is not None:
        print(f"warm mean: {warm_mean:.2f} tok/s")
        print(f"warm stddev: {warm_std:.2f} tok/s" if warm_std is not None else "warm stddev: n/a")
    print(sampler.report())
    cpu_m, _ = sampler.mean_std(sampler.cpu)
    gpu_m, _ = sampler.mean_std(sampler.gpu)
    vram_m, _ = sampler.mean_std(sampler.vram_mb)
    payload = {
        "cold_tps": cold,
        "warm_tps": warm_mean,
        "warm_sd": warm_std,
        "cpu_warm": cpu_m,
        "gpu_warm": gpu_m,
        "vram_warm": round(vram_m / 1024, 2) if vram_m else None,
    }
    print(f"JSON: {json.dumps(payload)}")


# ---- models (input/output contract for run-all + analyze) ----


class Backend(StrEnum):
    """Which engine runs the series."""

    LLAMA = "llama"
    COLIBRI = "colibri"


class Mode(StrEnum):
    """How the series is executed."""

    SERVE = "serve"
    STANDALONE = "standalone"


class Container(StrEnum):
    """colibri quant container variant."""

    INT4 = "int4"
    GS64 = "gs64"


class Run(BaseModel):
    """
    One benchmark series: the input definition for run-all.

    Rings through as-is: name labels the bench.log segment and the table.md row.
    """

    name: str  # "<domain> | <backend>", e.g. "math | llama"
    backend: Backend
    mode: Mode
    model: str
    prompts: Path
    container: Container = Container.INT4
    snap: Path | None = None
    expert_gb: str = "5"
    max_tokens: int = 100
    port: int = 8888
    heat: Path | None = None

    @property
    def log_header(self) -> str:
        """Run marker written verbatim into bench.log (analyze greps for it)."""
        return f"[{self.name}]"


class Probe(BaseModel):
    """Metrics of one standalone engine process, parsed from its stderr."""

    speed: float | None = None
    ttft: float | None = None
    vhit: float | None = None
    cmiss: float | None = None
    rhit: float | None = None


class SeriesMetrics(BaseModel):
    """
    Output contract: exactly the JSON every run emits as its last 'JSON:' line.

    Runs dump these fields directly; analyze validates them back into this model
    and table.md renders each as one row (None -> '-').
    """

    name: str
    cold_tps: float | None = None
    cold_ttft: float | None = None
    cold_vhit: float | None = None
    warm_tps: float | None = None
    warm_sd: float | None = None
    warm_vhit: float | None = None
    warm_vhit_sd: float | None = None
    cpu_warm: float | None = None
    gpu_warm: float | None = None
    vram_warm: float | None = None


# ---- analyze (bench.log -> Markdown table) ----


def _fmt(v: float | None) -> str:
    """Format one metric cell; None renders as '-'."""
    return "-" if v is None else f"{v:.2f}"


def _row(m: SeriesMetrics) -> str:
    """Render one table.md line from a SeriesMetrics row."""
    domain, backend = m.name.split(" | ", 1)
    cells = [
        domain,
        backend,
        _fmt(m.cold_tps),
        f"{_fmt(m.warm_tps)} ({_fmt(m.warm_sd)})",
        _fmt(m.cold_vhit),
        f"{_fmt(m.warm_vhit)} ({_fmt(m.warm_vhit_sd)})",
        _fmt(m.cpu_warm),
        _fmt(m.gpu_warm),
        _fmt(m.vram_warm),
    ]
    return "| " + " | ".join(cells) + " |"


def _table(metrics: list[SeriesMetrics]) -> str:
    """Render all measured runs as a single Markdown table."""
    header = (
        "| Domäne | Backend | cold tok/s | warm tok/s ± | cold VRAMhit | warm VRAMhit ± | "
        "CPU warm | GPU warm | VRAM warm |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    return "\n".join([header, *(_row(m) for m in metrics)])


def _json_line(line: str) -> dict[str, object] | None:
    """Parse a 'JSON: {...}' line, None if malformed."""
    try:
        return json.loads(line.removeprefix("JSON: "))
    except json.JSONDecodeError:
        return None


def cmd_analyze(args: argparse.Namespace) -> None:
    """Render the JSON lines of results/bench.log as a Markdown table."""
    name = ""
    rows: list[SeriesMetrics] = []
    for line in Path(args.log).read_text().splitlines():
        if line.startswith("["):
            name = line.strip("[]")
        elif (data := _json_line(line)) is not None:
            rows.append(SeriesMetrics.model_validate({"name": name, **data}))
    table = _table(rows)
    Path(args.out).write_text(table + "\n")
    print(table)


# ---- run-all (orchestrate all series -> results/bench.log) ----


def _kill_server() -> None:
    """Kill leftover servers so the next series' resource samples stay clean."""
    for pat in ("qwen36", "coli serve", "llama-server"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(5)


def _captured(func: Callable[[argparse.Namespace], None], namespace: argparse.Namespace) -> str:
    """Run func in-process, return merged stdout+stderr as a string."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        func(namespace)
    return buf.getvalue()


def _default_runs(domains: list[str], backends: list[str]) -> list[Run]:
    """Build the standard battery: every domain x backend as a declared Run."""
    return [
        Run(
            name=f"{domain} | {backend}",
            backend=Backend.LLAMA if backend == "llama" else Backend.COLIBRI,
            mode=Mode.SERVE if backend == "llama" else Mode.STANDALONE,
            model="qwen3.6-35b-a3b" if backend == "llama" else "qwen3.6-colibri",
            prompts=REPO / f"prompts/{domain}.txt",
            container=Container.INT4 if backend == "colibri-int4" else Container.GS64,
            snap=None if backend == "llama" else (COLIBRI_GS64 if backend == "colibri-gs64" else COLIBRI_INT4),
            heat=None if backend == "llama" else Path(f"/tmp/q36_{domain}_{backend}.heat"),
        )
        for domain in domains
        for backend in backends
    ]


def _series_run(run: Run) -> str:
    """
    Run one series, return captured output.

    llama stays in-process (serve is a plain HTTP client); colibri shells through
    nix-shell because qwen36.real links libcudart/libcublas that only exist there.
    """
    print(f">>> {run.log_header} ({time.strftime('%H:%M:%S')})")
    _kill_server()
    if run.mode is Mode.SERVE:
        cmd_start(
            argparse.Namespace(
                backend=run.backend,
                port=run.port,
                container=run.container,
                snap=run.snap,
                expert_gb=run.expert_gb,
            )
        )
        time.sleep(LLAMA_LOAD)
        out = _captured(
            cmd_serve,
            argparse.Namespace(
                url=f"http://127.0.0.1:{run.port}/v1/chat/completions",
                model=run.model,
                prompts=str(run.prompts),
                max_tokens=run.max_tokens,
                mode=run.backend,
            ),
        )
    else:
        snap = run.snap or COLIBRI_INT4
        inner = (
            f"export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1; "
            f"cd {REPO} && uv run bench standalone --prompts {run.prompts} --snap {snap} "
            f"--heat {run.heat} --expert-gb {run.expert_gb}"
        )
        r = subprocess.run(
            [
                "bash",
                "-lc",
                (
                    "nix-shell -p cudaPackages_12.cuda_cudart cudaPackages_12.cuda_cccl "
                    f"cudaPackages_12.libcublas gmp --run '{inner}'"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        out = r.stdout + r.stderr
    _kill_server()
    return out


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run all series from a run list, raw output -> results/bench.log."""
    log = REPO / "results" / "bench.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    runs = _default_runs(args.domains.split(","), args.backends.split(","))
    if args.runs:
        try:
            runs = [Run(**r) for r in json.loads(Path(args.runs).read_text())]
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            sys.exit(f"--runs {args.runs}: {e}")
    log.write_text(
        f"=== llm-bench results ({time.strftime('%Y-%m-%dT%H:%M:%S')}) ===\n"
        "Protocol: Q1-10 EN, Q11-20 DE, Q21-30 FR. "
        "colibri: Q1=cold, Q2-20=heat, Q21-30=warm (standalone, VRAM hit rate). "
        "llama: Q1-30=series (serve).\n"
        "GTX 1660 Ti / 6GB / sm_75, CUDA_EXPERT_GB=5\n"
        f"Backends: {args.backends}\n\n"
    )
    for run in runs:
        with log.open("a") as fh:
            fh.write(f"\n{run.log_header}\n{_series_run(run)}\n")
    _kill_server()
    with log.open("a") as fh:
        fh.write(f"=== done {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
    print(f">>> ALL DONE — {len(runs)} runs → {log}")


# ---- standalone (fresh process per prompt, colibri only) ----


def _run_once(
    engine: str, snap: str, expert_gb: str, n_new: str, heat: Path, prompt: str, *, cuda: bool = True
) -> Probe:
    """Run one standalone process, return parsed metrics as a typed Probe."""
    pf = Path(tempfile.mkstemp(suffix=".txt")[1])
    pf.write_text(prompt + "\n")
    env = {
        **os.environ,
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "SNAP": snap,
        "COLI_CUDA": "1" if cuda else "0",
        "COLI_GPUS": "0",
        "CUDA_EXPERT_GB": expert_gb,
        "N_NEW": n_new,
        "HEAT_FILE": str(heat),
    }
    r = subprocess.run(
        [engine, "256", "4", str(pf)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=600,
    )
    pf.unlink()
    raw = {
        "speed": _grep(r"^Speed:\s*([\d.]+)", r.stderr, multiline=True),
        "ttft": _grep(r"TTFT:\s*([\d.]+)", r.stderr),
        "vhit": _grep(r"VRAM hit rate:\s*([\d.]+)", r.stderr),
        "cmiss": _grep(r"miss\(CPU\)\s*(\d+)", r.stderr),
        "rhit": _grep(r"Expert cache hit rate:\s*([\d.]+)", r.stderr),
    }
    clean: dict[str, str | None] = {k: (v if v != "-" else None) for k, v in raw.items()}
    return Probe.model_validate(clean)


def cmd_standalone(args: argparse.Namespace) -> None:
    """Run colibri standalone: fresh process per prompt, heat accumulates over Q1-Q30."""
    prompts = Path(args.prompts).read_text().strip().splitlines()
    if len(prompts) < N_PROMPTS:
        msg = f"need {N_PROMPTS} prompts, got {len(prompts)}"
        raise SystemExit(msg)

    heat = Path(args.heat)
    if heat.exists():
        heat.unlink()  # Q1 = cold: start without accumulated heat

    cold: Probe | None = None
    warm_speed: list[float] = []
    warm_vhit: list[float] = []
    sampler = ResourceSampler()
    sampler_started = False

    for i, p in enumerate(prompts):
        m = _run_once(args.engine, args.snap, args.expert_gb, args.n_new, heat, p, cuda=args.cuda)
        if i == 0:
            cold = m
            print(f"cold: req{i} {_fmt(m.speed)} tok/s TTFT {_fmt(m.ttft)}s VRAMhit {_fmt(m.vhit)}%")
        elif 1 <= i <= HEAT_LAST:
            print(f"heat: req{i} {_fmt(m.speed)} tok/s (discarded)")
        else:
            if not sampler_started:  # sample resources only during warm (Q21-30)
                sampler.start()
                sampler_started = True
            if m.speed is not None:
                warm_speed.append(m.speed)
            if m.vhit is not None:
                warm_vhit.append(m.vhit)
            print(f"warm: req{i} {_fmt(m.speed)} tok/s TTFT {_fmt(m.ttft)}s VRAMhit {_fmt(m.vhit)}%")

    sampler.stop()
    print("---")
    if cold:
        print(f"cold: {_fmt(cold.speed)} tok/s TTFT {_fmt(cold.ttft)}s VRAMhit {_fmt(cold.vhit)}%")
    _mean_std(warm_speed, "warm tok/s mean")
    _mean_std(warm_vhit, "warm VRAMhit % mean")
    print(sampler.report())
    _push_summary("standalone", args.expert_gb, cold, warm_speed, warm_vhit, sampler)
    cpu_m, _ = sampler.mean_std(sampler.cpu)
    gpu_m, _ = sampler.mean_std(sampler.gpu)
    vram_m, _ = sampler.mean_std(sampler.vram_mb)
    payload = {
        "cold_tps": cold.speed if cold else None,
        "cold_ttft": cold.ttft if cold else None,
        "cold_vhit": cold.vhit if cold else None,
        "warm_tps": statistics.mean(warm_speed) if warm_speed else None,
        "warm_sd": statistics.stdev(warm_speed) if len(warm_speed) >= STDDEV_MIN else None,
        "warm_vhit": statistics.mean(warm_vhit) if warm_vhit else None,
        "warm_vhit_sd": statistics.stdev(warm_vhit) if len(warm_vhit) >= STDDEV_MIN else None,
        "cpu_warm": cpu_m,
        "gpu_warm": gpu_m,
        "vram_warm": round(vram_m / 1024, 2) if vram_m else None,
    }
    print(f"JSON: {json.dumps(payload)}")


def _push_summary(
    mode: str,
    expert_gb: str,
    cold: Probe | None,
    warm_speed: list[float],
    warm_vhit: list[float],
    sampler: ResourceSampler,
) -> None:
    """Send a compact ntfy summary of the finished measurement."""
    parts = [f"[{mode} gb={expert_gb}]"]
    if cold and cold.speed is not None:
        parts.append(f"cold {cold.speed:.2f} tok/s")
    if warm_speed:
        parts.append(f"warm {statistics.mean(warm_speed):.2f} tok/s")
    if warm_vhit:
        parts.append(f"hit {statistics.mean(warm_vhit):.1f}%")
    cpu_m, _ = sampler.mean_std(sampler.cpu)
    gpu_m, _ = sampler.mean_std(sampler.gpu)
    if cpu_m is not None:
        parts.append(f"CPU {cpu_m:.0f}%")
    if gpu_m is not None:
        parts.append(f"GPU {gpu_m:.0f}%")
    notify(" | ".join(parts))


def main() -> None:
    """Entry point: dispatch to start, serve, or standalone subcommand."""
    ap = argparse.ArgumentParser(description="LLM benchmark: serve + standalone")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_start = sub.add_parser("start", help="launch a server in background")
    sp_start.add_argument("--backend", choices=["colibri", "llama"], required=True)
    sp_start.add_argument("--port", type=int, default=8888)
    sp_start.add_argument("--expert-gb", default="5")
    sp_start.add_argument("--container", choices=["int4", "gs64"], default="int4")
    sp_start.add_argument("--snap", default=None)
    sp_start.set_defaults(func=cmd_start)

    sp_serve = sub.add_parser("serve", help="benchmark a running server")
    sp_serve.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
    sp_serve.add_argument("--model", default="qwen3.6-colibri")
    sp_serve.add_argument("--prompts", default="prompts/math.txt")
    sp_serve.add_argument("--max-tokens", type=int, default=100)
    sp_serve.add_argument("--mode", choices=["colibri", "llama"], default="colibri")
    sp_serve.set_defaults(func=cmd_serve)

    sp_standalone = sub.add_parser("standalone", help="colibri standalone, fresh process per prompt")
    sp_standalone.add_argument("--prompts", default="prompts/math.txt")
    sp_standalone.add_argument("--expert-gb", default="5")
    sp_standalone.add_argument("--engine", default=str(Path.home() / "src/colibri/c/qwen36.real"))
    sp_standalone.add_argument("--snap", default=str(Path.home() / "models/qwen36-35b-a3b-colibri-i4"))
    sp_standalone.add_argument("--n-new", default="100")
    sp_standalone.add_argument("--heat", default="/tmp/q36_standalone.heat")
    sp_standalone.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    sp_standalone.set_defaults(func=cmd_standalone)

    sp_all = sub.add_parser("run-all", help="orchestrate all series (6 domains x backends) -> results/bench.log")
    sp_all.add_argument("--port", type=int, default=8888)
    sp_all.add_argument("--domains", default=",".join(DOMAINS))
    sp_all.add_argument("--backends", default=",".join(BACKENDS))
    sp_all.add_argument("--runs", default=None, help="JSON list of Run definitions; overrides --domains/--backends")
    sp_all.set_defaults(func=cmd_run_all)

    sp_analyze = sub.add_parser("analyze", help="parse results/bench.log into a Markdown table")
    sp_analyze.add_argument("--log", default="results/bench.log")
    sp_analyze.add_argument("--out", default="results/table.md")
    sp_analyze.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
