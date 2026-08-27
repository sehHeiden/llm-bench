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
from pydantic_settings import BaseSettings, CliApp, CliSubCommand, SettingsConfigDict

from .notify import notify

N_PROMPTS = 30
HEAT_LAST = 19  # Q2-Q20 (index 1-19) build heat, discarded
STDDEV_MIN = 2
RETRY_MAX = 3
RETRY_WAIT = 10
HTTP_OK = 200
HTTP_SERVICE_UNAVAILABLE = 503


class Backend(StrEnum):
    """Which engine runs the series."""

    llama = "llama"
    colibri = "colibri"


class Container(StrEnum):
    """colibri quant container variant."""

    int4 = "int4"
    gs64 = "gs64"


COLIBRI_INT4 = Path.home() / "models/qwen36-35b-a3b-colibri-i4"
COLIBRI_GS64 = Path.home() / "models/qwen36-35b-a3b-colibri-i4-gs64"
LLAMA_WRAPPER = Path.home() / "bin/llama-server-qwen.sh"
REPO = Path(__file__).resolve().parents[2]
DOMAINS = ["math", "geography", "history", "philosophy", "physics", "chemistry"]
BACKENDS = ["llama", "colibri-int4", "colibri-gs64"]
LLAMA_LOAD = 45  # seconds for the llama model to load before series start
NIX_CUDA = "nix-shell -p cudaPackages_12.cuda_cudart cudaPackages_12.cuda_cccl cudaPackages_12.libcublas"


class ResourceSampler:
    """Background thread sampling CPU% and GPU%+VRAM during bench."""

    _NSMI = ("nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits")

    def __init__(self) -> None:
        """Initialize empty sample lists and stop event."""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu: list[float] = []
        self.gpu: list[float] = []
        self.vram_mb: list[float] = []

    def start(self) -> None:
        """Start the sampler thread; idempotent."""
        if self._thread and self._thread.is_alive():
            return
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
            try:
                out = subprocess.run(self._NSMI, capture_output=True, text=True, check=False, timeout=5).stdout
                gpu, vram = (float(x) for x in out.strip().split(", "))
                self.gpu.append(gpu)
                self.vram_mb.append(vram)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                pass  # no GPU / malformed output: skip sample
            self._stop.wait(1.0)

    def report(self) -> str:
        """Format mean+stddev lines for CPU, GPU, VRAM."""
        lines = []
        for label, values, div, unit in (
            ("CPU", self.cpu, 1, "%"),
            ("GPU", self.gpu, 1, "%"),
            ("VRAM", self.vram_mb, 1024, " GB"),
        ):
            m, s = _mean_std(values)
            if m is None:
                continue
            line = f"{label} mean: {m / div:.2f}{unit}, n={len(values)}"
            if s is not None:
                line += f" (stddev {s / div:.2f}{unit})"
            lines.append(line)
        return "\n".join(lines) if lines else "no samples"

    def stats(self) -> tuple[float | None, float | None, float | None]:
        """(cpu_warm, gpu_warm, vram_warm in GB)."""
        cpu_m, _ = _mean_std(self.cpu)
        gpu_m, _ = _mean_std(self.gpu)
        vram_m, _ = _mean_std(self.vram_mb)
        return cpu_m, gpu_m, round(vram_m / 1024, 2) if vram_m else None


def _grep(pattern: str, text: str, *, multiline: bool = False) -> str | None:
    """First regex group match or None."""
    if m := re.search(pattern, text, re.MULTILINE if multiline else 0):
        return m.group(1)
    return None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """(mean, stddev), stddev 0.0 until n >= STDDEV_MIN; (None, None) if empty."""
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) >= STDDEV_MIN else 0.0


def _phase(i: int, *, all_warm: bool = False) -> str:
    """Q protocol phase: 0=cold, 1..HEAT_LAST=heat (discarded), rest=warm."""
    if all_warm or i > HEAT_LAST:
        return "warm"
    return "cold" if i == 0 else "heat"


# ---- CLI (pydantic-settings subcommands, replaces argparse) ----


class StartArgs(BaseSettings):
    """Launch a server in background."""

    backend: Backend
    port: int = 8888
    expert_gb: str = "5"
    container: Container = Container.int4
    snap: Path | None = None

    def cli_cmd(self) -> None:
        """Dispatch to the command implementation."""
        cmd_start(self)


class ServeArgs(BaseSettings):
    """Benchmark a running server."""

    url: str = "http://127.0.0.1:8888/v1/chat/completions"
    model: str = "qwen3.6-colibri"
    prompts: Path = Path("prompts/math.txt")
    max_tokens: int = 100
    mode: Backend = Backend.colibri

    def cli_cmd(self) -> None:
        """Dispatch to the command implementation."""
        cmd_serve(self)


class StandaloneArgs(BaseSettings):
    """colibri standalone, fresh process per prompt."""

    prompts: Path = Path("prompts/math.txt")
    expert_gb: str = "5"
    engine: Path = Path.home() / "src/colibri/c/qwen36.real"
    snap: Path = Path.home() / "models/qwen36-35b-a3b-colibri-i4"
    n_new: str = "100"
    heat: Path = Path("/tmp/q36_standalone.heat")
    cuda: bool = True

    def cli_cmd(self) -> None:
        """Dispatch to the command implementation."""
        cmd_standalone(self)


class RunAllArgs(BaseSettings):
    """Orchestrate all series (6 domains x backends) -> results/bench.log."""

    domains: str = ",".join(DOMAINS)
    backends: str = ",".join(BACKENDS)
    runs: Path | None = None  # JSON list of Run definitions; overrides --domains/--backends

    def cli_cmd(self) -> None:
        """Dispatch to the command implementation."""
        cmd_run_all(self)


class AnalyzeArgs(BaseSettings):
    """Parse results/bench.log into a Markdown table."""

    log: Path = Path("results/bench.log")
    out: Path = Path("results/table.md")

    def cli_cmd(self) -> None:
        """Dispatch to the command implementation."""
        cmd_analyze(self)


class Bench(BaseSettings):
    """llm-bench CLI: bench <subcommand> [options]."""

    model_config = SettingsConfigDict(cli_kebab_case="all")  # enum choices lowercase: LLAMA -> llama

    start: CliSubCommand[StartArgs]
    serve: CliSubCommand[ServeArgs]
    standalone: CliSubCommand[StandaloneArgs]
    run_all: CliSubCommand[RunAllArgs]
    analyze: CliSubCommand[AnalyzeArgs]

    def cli_cmd(self) -> None:
        """Dispatch to the command implementation."""
        CliApp.run_subcommand(self)


def _load_prompts(path: Path) -> list[str]:
    """Read the prompts file; enforce the 30-question protocol length."""
    prompts = path.read_text().strip().splitlines()
    if len(prompts) < N_PROMPTS:
        raise SystemExit(f"need {N_PROMPTS} prompts, got {len(prompts)}")
    return prompts


def _spawn(log: Path, argv: list[str], env: dict[str, str] | None = None) -> None:
    """Start a detached background process, output merged into log."""
    with log.open("w") as fh:
        subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, env=env)


# ---- serve (persistent server) ----


def _retry(e: Exception, idx: int, attempt: int) -> None:
    """Sleep for the next retry, or give up when retries are exhausted."""
    if attempt >= RETRY_MAX - 1:
        raise e
    print(f"  [req {idx}] {e}, retry {attempt + 1}/{RETRY_MAX} in {RETRY_WAIT}s...")
    time.sleep(RETRY_WAIT)


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
            _retry(e, idx, attempt)
            continue
        except urllib.error.URLError as e:
            _retry(e, idx, attempt)
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


def cmd_start(args: StartArgs) -> None:
    """Launch a server (colibri or llama) in background via nix-shell."""
    port = args.port
    if args.backend is Backend.llama:
        log = Path(f"/tmp/llama_serve_{port}.log")
        _spawn(log, [str(LLAMA_WRAPPER)])
        print(f"started llama serve on port {port} (log {log})")
    else:  # colibri
        snap = str(args.snap or (COLIBRI_GS64 if args.container is Container.gs64 else COLIBRI_INT4))
        log = Path(f"/tmp/q36_serve_{port}.log")
        _spawn(
            log,
            [
                "bash",
                "-c",
                (
                    f"{NIX_CUDA} python3 gmp --run 'cd ~/src/colibri/c && COLI_CUDA=1 COLI_GPUS=0 "
                    f"CUDA_EXPERT_GB={args.expert_gb} HEAT_FILE=/tmp/q36_serve.heat COLI_MODEL={snap} "
                    f"python3 ./coli serve --model {snap} --cap 256 --port {port}'"
                ),
            ],
            env={**os.environ, "HOME": str(Path.home())},
        )
        print(f"started colibri serve ({args.container}) on port {port} (log {log})")
    print(f"waiting for port {port}...")
    if _wait_for_port(port):
        print(f"✓ port {port} listening")
    else:
        print(f"✗ port {port} not up after 180s — check {log}")


def cmd_serve(args: ServeArgs) -> None:
    """Benchmark a running server (colibri or llama), Q1/Q2-20/Q21-30 protocol."""
    prompts = _load_prompts(args.prompts)
    cold: float | None = None
    warm: list[float] = []
    sampler = ResourceSampler()

    for i, p in enumerate(prompts):
        tps = _request(args.url, args.model, p, args.max_tokens, i)
        ph = _phase(i, all_warm=args.mode is Backend.llama)
        if ph == "warm":
            sampler.start()
            warm.append(tps)
        elif ph == "cold" and cold is None:
            cold = tps
        print(f"{ph}: req{i} {tps:.2f} tok/s" + (" (discarded)" if ph == "heat" else ""))

    sampler.stop()
    print("---")
    if cold is not None:
        print(f"cold: {cold:.2f} tok/s")
    warm_mean, warm_std = _mean_std(warm)
    if warm_mean is not None:
        print(f"warm mean: {warm_mean:.2f} tok/s")
        print(f"warm stddev: {warm_std:.2f} tok/s" if len(warm) >= STDDEV_MIN else "warm stddev: n/a")
    print(sampler.report())
    cpu_warm, gpu_warm, vram_warm = sampler.stats()
    metrics = SeriesMetrics(
        cold_tps=cold,
        warm_tps=warm_mean,
        warm_sd=warm_std if len(warm) >= STDDEV_MIN else None,
        cpu_warm=cpu_warm,
        gpu_warm=gpu_warm,
        vram_warm=vram_warm,
    )
    print(f"JSON: {metrics.model_dump_json()}")


# ---- models (input/output contract for run-all + analyze) ----


class Run(BaseModel):
    """
    One benchmark series: the input definition for run-all.

    Rings through as-is: name labels the bench.log segment and the table.md row.
    """

    name: str  # "<domain> | <backend>", e.g. "math | llama"
    backend: Backend
    model: str
    prompts: Path
    container: Container = Container.int4
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


class SeriesMetrics(BaseModel):
    """
    Output contract: exactly the JSON every run emits as its last 'JSON:' line.

    Runs dump these fields directly; analyze validates them back into this model
    and table.md renders each as one row (None -> '-').
    """

    name: str = ""
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


def cmd_analyze(args: AnalyzeArgs) -> None:
    """Render the JSON lines of results/bench.log as a Markdown table."""
    name = ""
    rows: list[SeriesMetrics] = []
    for line in args.log.read_text().splitlines():
        if line.startswith("["):
            name = line.strip("[]")
        elif (data := _json_line(line)) is not None:
            rows.append(SeriesMetrics.model_validate({"name": name, **data}))
    table = _table(rows)
    args.out.write_text(table + "\n")
    print(table)


# ---- run-all (orchestrate all series -> results/bench.log) ----


def _kill_server() -> None:
    """Kill leftover servers so the next series' resource samples stay clean."""
    for pat in ("qwen36", "coli serve", "llama-server"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(5)


def _captured(func: Callable[[ServeArgs], None], args: ServeArgs) -> str:
    """Run func in-process, return merged stdout+stderr as a string."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        func(args)
    return buf.getvalue()


def _default_runs(domains: list[str], backends: list[str]) -> list[Run]:
    """Build the standard battery: every domain x backend as a declared Run."""
    return [
        Run(
            name=f"{domain} | {backend}",
            backend=Backend.llama if backend == "llama" else Backend.colibri,
            model="qwen3.6-35b-a3b" if backend == "llama" else "qwen3.6-colibri",
            prompts=REPO / f"prompts/{domain}.txt",
            container=Container.int4 if backend == "colibri-int4" else Container.gs64,
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
    if run.backend is Backend.llama:
        cmd_start(
            StartArgs(
                backend=run.backend, port=run.port, container=run.container, snap=run.snap, expert_gb=run.expert_gb
            )
        )
        time.sleep(LLAMA_LOAD)
        out = _captured(
            cmd_serve,
            ServeArgs(
                url=f"http://127.0.0.1:{run.port}/v1/chat/completions",
                model=run.model,
                prompts=run.prompts,
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
            ["bash", "-lc", f"{NIX_CUDA} gmp --run '{inner}'"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = r.stdout + r.stderr
    _kill_server()
    return out


def cmd_run_all(args: RunAllArgs) -> None:
    """Run all series from a run list, raw output -> results/bench.log."""
    log = REPO / "results" / "bench.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    runs = _default_runs(args.domains.split(","), args.backends.split(","))
    if args.runs:
        try:
            runs = [Run(**r) for r in json.loads(args.runs.read_text())]
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


def _run_once(args: StandaloneArgs, heat: Path, prompt: str) -> Probe:
    """Run one standalone process, return parsed metrics as a typed Probe."""
    pf = Path(tempfile.mkstemp(suffix=".txt")[1])
    pf.write_text(prompt + "\n")
    env = {
        **os.environ,
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "SNAP": str(args.snap),
        "COLI_CUDA": "1" if args.cuda else "0",
        "COLI_GPUS": "0",
        "CUDA_EXPERT_GB": args.expert_gb,
        "N_NEW": args.n_new,
        "HEAT_FILE": str(heat),
    }
    r = subprocess.run(
        [str(args.engine), "256", "4", str(pf)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=600,
    )
    pf.unlink()
    return Probe.model_validate(
        {
            "speed": _grep(r"^Speed:\s*([\d.]+)", r.stderr, multiline=True),
            "ttft": _grep(r"TTFT:\s*([\d.]+)", r.stderr),
            "vhit": _grep(r"VRAM hit rate:\s*([\d.]+)", r.stderr),
        }
    )


def cmd_standalone(args: StandaloneArgs) -> None:
    """Run colibri standalone: fresh process per prompt, heat accumulates over Q1-Q30."""
    prompts = _load_prompts(args.prompts)
    heat = args.heat
    if heat.exists():
        heat.unlink()  # Q1 = cold: start without accumulated heat

    cold: Probe | None = None
    warm_speed: list[float] = []
    warm_vhit: list[float] = []
    sampler = ResourceSampler()

    for i, p in enumerate(prompts):
        m = _run_once(args, heat, p)
        ph = _phase(i)
        if ph == "cold":
            cold = m
        elif ph == "warm":
            sampler.start()
            if m.speed is not None:
                warm_speed.append(m.speed)
            if m.vhit is not None:
                warm_vhit.append(m.vhit)
        tail = " (discarded)" if ph == "heat" else f" TTFT {_fmt(m.ttft)}s VRAMhit {_fmt(m.vhit)}%"
        print(f"{ph}: req{i} {_fmt(m.speed)} tok/s{tail}")

    sampler.stop()
    print("---")
    if cold:
        print(f"cold: {_fmt(cold.speed)} tok/s TTFT {_fmt(cold.ttft)}s VRAMhit {_fmt(cold.vhit)}%")
    for label, vals in (("warm tok/s", warm_speed), ("warm VRAMhit %", warm_vhit)):
        m, s = _mean_std(vals)
        sd = f" (stddev {s:.2f})" if m is not None and len(vals) >= STDDEV_MIN else ""
        print(f"{label} mean: {m:.2f}{sd}" if m is not None else f"{label} mean: n/a")
    print(sampler.report())
    _push_summary(args, cold, warm_speed, warm_vhit, sampler)
    warm_mean, warm_sd = _mean_std(warm_speed)
    vhit_mean, vhit_sd = _mean_std(warm_vhit)
    cpu_warm, gpu_warm, vram_warm = sampler.stats()
    metrics = SeriesMetrics(
        cold_tps=cold.speed if cold else None,
        cold_ttft=cold.ttft if cold else None,
        cold_vhit=cold.vhit if cold else None,
        warm_tps=warm_mean,
        warm_sd=warm_sd if len(warm_speed) >= STDDEV_MIN else None,
        warm_vhit=vhit_mean,
        warm_vhit_sd=vhit_sd if len(warm_vhit) >= STDDEV_MIN else None,
        cpu_warm=cpu_warm,
        gpu_warm=gpu_warm,
        vram_warm=vram_warm,
    )
    print(f"JSON: {metrics.model_dump_json()}")


def _push_summary(
    args: StandaloneArgs,
    cold: Probe | None,
    warm_speed: list[float],
    warm_vhit: list[float],
    sampler: ResourceSampler,
) -> None:
    """Send a compact ntfy summary of the finished measurement."""
    parts = [f"[standalone gb={args.expert_gb}]"]
    if cold and cold.speed is not None:
        parts.append(f"cold {cold.speed:.2f} tok/s")
    if warm_speed:
        parts.append(f"warm {statistics.mean(warm_speed):.2f} tok/s")
    if warm_vhit:
        parts.append(f"hit {statistics.mean(warm_vhit):.1f}%")
    for label, values in (("CPU", sampler.cpu), ("GPU", sampler.gpu)):
        m, _ = _mean_std(values)
        if m is not None:
            parts.append(f"{label} {m:.0f}%")
    notify(" | ".join(parts))


def main() -> None:
    """Entry point: dispatch to the selected subcommand."""
    CliApp.run(Bench)


if __name__ == "__main__":
    main()
