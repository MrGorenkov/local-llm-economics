"""Energy and throughput of llama.cpp inference, measured per request on a running llama-server.

The model is loaded once. For every HTTP request we read the NVML hardware energy
counter right before sending and right after receiving the response, so model
loading and warm-up never enter the measurement. A 20 Hz power/CPU trace runs in
parallel as a cross-check.

Phases:
  pp  - prefill: a fixed ~512-token Russian prompt, n_predict = 1, prompt cache off
  tg  - decode: a short prompt, 256 generated tokens, EOS ignored, greedy

Usage:
  python bench_server.py --model M.gguf --ngl 99 --tag qwen3.5-2b/Q4_K_M
Appends one JSON line per request to results/energy_server.jsonl.
"""
import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pynvml as nvml

ROOT = Path(__file__).resolve().parents[1]
SERVER = Path(r"D:\papers-tools\llama.cpp-b11136\bin\llama-server.exe")
PORT = 8089
URL = f"http://127.0.0.1:{PORT}"
PP_TOKENS, TG_TOKENS = 512, 256
PROMPT_TEXT = (ROOT / "data" / "pp_prompt_ru.txt")
LOG_DIR = ROOT / "results"
GPU_INDEX = 0


class Sampler(threading.Thread):
    def __init__(self, handle, hz=20):
        super().__init__(daemon=True)
        self.h, self.dt = handle, 1.0 / hz
        self.rows, self.halt = [], threading.Event()
        psutil.cpu_percent(None)

    def run(self):
        while not self.halt.is_set():
            t = time.perf_counter()
            self.rows.append((t, nvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
                              nvml.nvmlDeviceGetMemoryInfo(self.h).used / 2**20,
                              nvml.nvmlDeviceGetUtilizationRates(self.h).gpu,
                              psutil.cpu_percent(None)))
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t)))

    def stop(self):
        self.halt.set()
        self.join()

    def window(self, t0, t1):
        return [r for r in self.rows if t0 <= r[0] <= t1]


def post(path, payload, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(proc, timeout=180):
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(URL + "/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def gpu_process_memory(pid):
    """Dedicated and shared (system-memory fallback) GPU memory of a process, MiB. Windows only."""
    if os.name != "nt":
        return None, None
    ps = (f"$c = Get-Counter -Counter '\\GPU Process Memory(pid_{pid}*)\\Dedicated Usage',"
          f"'\\GPU Process Memory(pid_{pid}*)\\Shared Usage' -ErrorAction SilentlyContinue; "
          "$d = ($c.CounterSamples | Where-Object {$_.Path -match 'dedicated'} | Measure-Object CookedValue -Sum).Sum; "
          "$s = ($c.CounterSamples | Where-Object {$_.Path -match 'shared'} | Measure-Object CookedValue -Sum).Sum; "
          "Write-Output \"$d;$s\"")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    try:
        d, s = r.stdout.strip().split(";")
        return float(d) / 2**20, float(s) / 2**20
    except ValueError:
        return None, None


def start_server(model, ngl, threads, ctx):
    """ngl < 0 means: do not pass -ngl and let llama.cpp --fit choose the layer split."""
    args = [str(SERVER), "-m", str(model), "-t", str(threads),
            "-c", str(ctx), "-np", "1", "--port", str(PORT), "--host", "127.0.0.1",
            "--no-webui"]
    if ngl >= 0:
        args += ["-ngl", str(ngl)]
    log = open(LOG_DIR / "server_last.log", "w", encoding="utf-8")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    if not wait_ready(proc):
        proc.kill()
        log.close()
        return None
    return proc


def fixed_prompt(n_tokens):
    """Russian text trimmed to exactly n_tokens model tokens (via the server tokenizer)."""
    text = PROMPT_TEXT.read_text(encoding="utf-8")
    toks = post("/tokenize", {"content": text})["tokens"]
    if len(toks) < n_tokens:
        raise SystemExit("prompt source too short")
    return post("/detokenize", {"tokens": toks[:n_tokens]})["content"]


MIN_WINDOW_S = 5.0  # NVML counters update every ~100 ms; short windows are too noisy


def measure(h, sampler, payload, phase, cpu_meter=None):
    """Send identical requests back to back until the window lasts >= MIN_WINDOW_S."""
    e0 = nvml.nvmlDeviceGetTotalEnergyConsumption(h)
    c0 = cpu_meter.energy_j() if cpu_meter else None
    t0 = time.perf_counter()
    timings = []
    while True:
        timings.append(post("/completion", payload)["timings"])
        if time.perf_counter() - t0 >= MIN_WINDOW_S:
            break
    t1 = time.perf_counter()
    e1 = nvml.nvmlDeviceGetTotalEnergyConsumption(h)
    c1 = cpu_meter.energy_j() if cpu_meter else None
    n = sum(tm["prompt_n"] if phase == "pp" else tm["predicted_n"] for tm in timings)
    ms = sum(tm["prompt_ms"] if phase == "pp" else tm["predicted_ms"] for tm in timings)
    win = sampler.window(t0, t1)
    trace = sum((win[i + 1][0] - win[i][0]) * win[i][1] for i in range(len(win) - 1))
    return {
        "energy_j": (e1 - e0) / 1000.0, "energy_trace_j": trace, "wall_s": t1 - t0,
        "cpu_energy_j": (c1 - c0) if cpu_meter else None,
        "requests": len(timings), "tokens": n,
        "prompt_n": sum(tm["prompt_n"] for tm in timings),
        "predicted_n": sum(tm["predicted_n"] for tm in timings),
        "prompt_tps": (sum(tm["prompt_n"] for tm in timings) /
                       (sum(tm["prompt_ms"] for tm in timings) / 1000.0)),
        "predicted_tps": (sum(tm["predicted_n"] for tm in timings) /
                          max(1e-9, sum(tm["predicted_ms"] for tm in timings) / 1000.0)),
        "phase_ms": ms,
        "peak_vram_mib": max((r[2] for r in win), default=None),
        "mean_gpu_util": sum(r[3] for r in win) / len(win) if win else None,
        "mean_cpu_util": sum(r[4] for r in win) / len(win) if win else None,
        "mean_power_w": sum(r[1] for r in win) / len(win) if win else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "energy_server.jsonl"))
    ap.add_argument("--server", default=str(SERVER))
    ap.add_argument("--prompt-file", default=str(PROMPT_TEXT))
    ap.add_argument("--log-dir", default=str(LOG_DIR))
    ap.add_argument("--gpu-index", type=int, default=GPU_INDEX)
    a = ap.parse_args()
    globals().update(SERVER=Path(a.server), PROMPT_TEXT=Path(a.prompt_file), LOG_DIR=Path(a.log_dir))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    nvml.nvmlInit()
    h = nvml.nvmlDeviceGetHandleByIndex(a.gpu_index)
    base = Sampler(h)
    base.start()
    time.sleep(5)
    base.stop()
    idle_empty = sum(r[1] for r in base.rows) / len(base.rows)

    proc = start_server(a.model, a.ngl, a.threads, a.ctx)
    if proc is None:
        print(f"[{a.tag} ngl={a.ngl}] server failed to start (likely out of VRAM)")
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tag": a.tag, "model": Path(a.model).name, "ngl": a.ngl,
                                "status": "failed_to_start"}) + "\n")
        return 2
    s = Sampler(h)
    s.start()
    try:
        prompt_pp = fixed_prompt(PP_TOKENS)
        pp_payload = {"prompt": prompt_pp, "n_predict": 1, "cache_prompt": False,
                      "temperature": 0.0}
        tg_payload = {"prompt": "Расскажи подробно об истории Москвы.", "n_predict": TG_TOKENS,
                      "ignore_eos": True, "cache_prompt": False, "temperature": 0.0}
        for _ in range(2):  # warm-up, not recorded
            post("/completion", pp_payload)
            post("/completion", {**tg_payload, "n_predict": 32})
        dedicated_mib, shared_mib = gpu_process_memory(proc.pid)
        time.sleep(3)
        t_idle0 = time.perf_counter()
        time.sleep(5)
        idle_loaded = s.window(t_idle0, time.perf_counter())
        idle_loaded_w = sum(r[1] for r in idle_loaded) / len(idle_loaded)
        recs = []
        for i in range(a.reps):
            for phase, payload in (("pp", pp_payload), ("tg", tg_payload)):
                m = measure(h, s, payload, phase)
                m.update({
                    "time": datetime.now(timezone.utc).isoformat(), "tag": a.tag,
                    "model": Path(a.model).name, "ngl": a.ngl, "threads": a.threads,
                    "ctx": a.ctx, "phase": phase, "rep": i,
                    "idle_empty_w": idle_empty, "idle_loaded_w": idle_loaded_w,
                    "proc_dedicated_mib": dedicated_mib, "proc_shared_mib": shared_mib,
                    "j_per_token": m["energy_j"] / m["tokens"],
                    "net_j_per_token": (m["energy_j"] - idle_loaded_w * m["wall_s"]) / m["tokens"],
                    "status": "ok",
                })
                recs.append(m)
                time.sleep(1)
        with open(a.out, "a", encoding="utf-8") as f:
            for m in recs:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        for phase in ("pp", "tg"):
            r = [m for m in recs if m["phase"] == phase]
            jpt = sorted(m["j_per_token"] for m in r)
            tps = [m["prompt_tps"] if phase == "pp" else m["predicted_tps"] for m in r]
            print(f"[{a.tag} ngl={a.ngl} {phase}] median {jpt[len(jpt) // 2]:.4f} J/tok "
                  f"(min {jpt[0]:.4f}, max {jpt[-1]:.4f}); {sum(tps) / len(tps):.1f} tok/s; "
                  f"power {sum(m['mean_power_w'] for m in r) / len(r):.1f} W; "
                  f"VRAM {max(m['peak_vram_mib'] for m in r):.0f} MiB; "
                  f"CPU {sum(m['mean_cpu_util'] for m in r) / len(r):.0f}%; "
                  f"trace/counter {sum(m['energy_trace_j'] for m in r) / sum(m['energy_j'] for m in r):.3f}")
        print(f"idle GPU power: empty {idle_empty:.1f} W, model loaded {idle_loaded_w:.1f} W")
    finally:
        s.stop()
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        nvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
