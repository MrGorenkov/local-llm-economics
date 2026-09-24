"""Local GTX 1650 campaign: per-request energy/throughput for every model x quantization.

For each GGUF: try full offload (ngl=99); if the server cannot start (out of VRAM),
step the number of GPU layers down until it fits. A separate sweep over ngl for
selected models records the "VRAM cliff". CPU-only (ngl=0) runs give a reference.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QDIR = Path(r"D:\papers-models\q")
PY = sys.executable
BENCH = ROOT / "code" / "bench_server.py"
OUT = ROOT / "results" / "energy_1650.jsonl"
MODELS = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"]
LADDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


def bench(model_path, ngl, tag, reps=5):
    r = subprocess.run([PY, str(BENCH), "--model", str(model_path), "--ngl", str(ngl), "--tag", tag,
                        "--reps", str(reps), "--out", str(OUT)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr[-500:], flush=True)
    return r.returncode == 0


def done(tag, ngl):
    if not OUT.exists():
        return False
    for line in OUT.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("tag") == tag and d.get("ngl") == ngl and d.get("status") == "ok":
            return True
    return False


def main(stage):
    if stage in ("all", "ladder"):
        for m in MODELS:
            for q in LADDER:
                path, tag = QDIR / m / f"{m}-{q}.gguf", f"{m}/{q}"
                for ngl in [99] + list(range(40, -1, -4)):
                    if done(tag, ngl):
                        break
                    if bench(path, ngl, tag):
                        break
    if stage in ("all", "cliff"):
        # Models that exceed 4 GB: compare (a) ngl=all with the Windows driver's system-memory
        # fallback (already in the ladder), (b) llama.cpp --fit auto split (ngl=-1),
        # (c) explicit CPU offload of the remaining layers. Q4_K_M 4B fits and is a reference.
        plan = {
            ("qwen3.5-4b", "F16"): (0, 4, 8, 12, 16),
            ("qwen3.5-4b", "Q8_0"): (0, 8, 12, 16, 20, 24),
            ("qwen3.5-4b", "Q6_K"): (16, 20, 24, 28),
            ("qwen3.5-2b", "F16"): (12, 16, 20, 24),
            ("qwen3.5-4b", "Q4_K_M"): (0, 8, 16, 24),
        }
        for (m, q), ngls in plan.items():
            path, tag = QDIR / m / f"{m}-{q}.gguf", f"cliff:{m}/{q}"
            for ngl in (-1,) + tuple(ngls):
                if not done(tag, ngl):
                    bench(path, ngl, tag, reps=3)
    if stage in ("all", "cpu"):
        for m in MODELS:
            path, tag = QDIR / m / f"{m}-Q4_K_M.gguf", f"cpu:{m}/Q4_K_M"
            if not done(tag, 0):
                bench(path, 0, tag, reps=3)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
