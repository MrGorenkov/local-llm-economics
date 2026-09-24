"""Energy and throughput of llama.cpp inference on the local NVIDIA GPU.

Method (differential): llama-bench is run twice for the same configuration, with
R_LO and R_HI timed repetitions. Model loading and warm-up cost the same in both
runs, so the energy of one repetition is (E_hi - E_lo) / (R_HI - R_LO), where E is
the delta of the NVML hardware energy counter over the whole process. A 20 Hz
power trace is recorded in parallel as a cross-check and for plots.

Prefill (pp) and decode (tg) are measured in separate runs so that energy is
attributed to one phase only.

Usage:
  python bench_energy.py --model M.gguf --ngl 99 --phase pp --tag qwen2b-q4km
Writes one JSON line per measurement pair to results/energy.jsonl.
"""
import argparse
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pynvml as nvml

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(r"D:\papers-tools\llama.cpp-b11136\bin\llama-bench.exe")
R_LO, R_HI = 2, 10
N_PROMPT, N_GEN = 512, 128


class Sampler(threading.Thread):
    """Samples GPU power, GPU memory and CPU utilisation at a fixed rate."""

    def __init__(self, handle, hz=20):
        super().__init__(daemon=True)
        self.h, self.dt = handle, 1.0 / hz
        self.rows, self.halt = [], threading.Event()
        psutil.cpu_percent(None)

    def run(self):
        while not self.halt.is_set():
            t = time.perf_counter()
            p = nvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
            mem = nvml.nvmlDeviceGetMemoryInfo(self.h).used / 2**20
            util = nvml.nvmlDeviceGetUtilizationRates(self.h).gpu
            cpu = psutil.cpu_percent(None)
            self.rows.append((t, p, mem, util, cpu))
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t)))

    def stop(self):
        self.halt.set()
        self.join()


def run_once(handle, model, ngl, phase, reps, threads):
    args = [str(BENCH), "-m", str(model), "-ngl", str(ngl), "-t", str(threads),
            "-r", str(reps), "-o", "json"]
    args += ["-p", str(N_PROMPT), "-n", "0"] if phase == "pp" else ["-p", "0", "-n", str(N_GEN)]
    s = Sampler(handle)
    e0 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    t0 = time.perf_counter()
    s.start()
    proc = subprocess.run(args, capture_output=True, text=True)
    s.stop()
    t1 = time.perf_counter()
    e1 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed (ngl={ngl}): {proc.stderr[-800:]}")
    bench = json.loads(proc.stdout)[0]
    rows = s.rows
    trace_j = sum((rows[i + 1][0] - rows[i][0]) * rows[i][1] for i in range(len(rows) - 1))
    return {
        "energy_counter_j": (e1 - e0) / 1000.0,
        "energy_trace_j": trace_j,
        "wall_s": t1 - t0,
        "peak_vram_mib": max(r[2] for r in rows) if rows else None,
        "mean_gpu_util": sum(r[3] for r in rows) / len(rows) if rows else None,
        "mean_cpu_util": sum(r[4] for r in rows) / len(rows) if rows else None,
        "bench": bench,
    }


def idle_power(handle, seconds=5.0):
    s = Sampler(handle)
    s.start()
    time.sleep(seconds)
    s.stop()
    return sum(r[1] for r in s.rows) / len(s.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--phase", choices=["pp", "tg"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=3, help="independent (R_LO, R_HI) pairs")
    ap.add_argument("--out", default=str(ROOT / "results" / "energy.jsonl"))
    a = ap.parse_args()

    nvml.nvmlInit()
    h = nvml.nvmlDeviceGetHandleByIndex(0)
    tokens = N_PROMPT if a.phase == "pp" else N_GEN
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for k in range(a.pairs):
        p_idle = idle_power(h)
        lo = run_once(h, a.model, a.ngl, a.phase, R_LO, a.threads)
        hi = run_once(h, a.model, a.ngl, a.phase, R_HI, a.threads)
        e_rep = (hi["energy_counter_j"] - lo["energy_counter_j"]) / (R_HI - R_LO)
        e_rep_trace = (hi["energy_trace_j"] - lo["energy_trace_j"]) / (R_HI - R_LO)
        t_rep = (hi["wall_s"] - lo["wall_s"]) / (R_HI - R_LO)
        rec = {
            "time": datetime.now(timezone.utc).isoformat(),
            "tag": a.tag, "model": Path(a.model).name, "ngl": a.ngl, "phase": a.phase,
            "threads": a.threads, "pair": k, "tokens_per_rep": tokens,
            "idle_power_w": p_idle,
            "energy_per_rep_j": e_rep,
            "energy_per_rep_trace_j": e_rep_trace,
            "j_per_token": e_rep / tokens,
            "net_j_per_token": (e_rep - p_idle * t_rep) / tokens,
            "wall_per_rep_s": t_rep,
            "tok_per_s_bench": hi["bench"].get("avg_ts"),
            "tok_per_s_bench_sd": hi["bench"].get("stddev_ts"),
            "peak_vram_mib": hi["peak_vram_mib"],
            "mean_gpu_util": hi["mean_gpu_util"],
            "mean_cpu_util": hi["mean_cpu_util"],
            "bench_meta": {k2: hi["bench"].get(k2) for k2 in
                           ("build_commit", "build_number", "cpu_info", "gpu_info", "backends",
                            "model_type", "model_size", "model_n_params", "n_gpu_layers",
                            "flash_attn", "type_k", "type_v", "n_batch", "n_ubatch")},
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{a.tag} {a.phase} ngl={a.ngl} pair={k}] "
              f"{rec['j_per_token']:.4f} J/tok (trace {e_rep_trace / tokens:.4f}), "
              f"{rec['tok_per_s_bench']:.1f} tok/s, VRAM {rec['peak_vram_mib']:.0f} MiB, "
              f"CPU {rec['mean_cpu_util']:.0f}%")
    nvml.nvmlShutdown()


if __name__ == "__main__":
    main()
