"""Where do the weights live? For every model x quantization, start llama-server with
ngl=all and record dedicated vs shared (driver system-memory fallback) GPU memory, and
the layer count that llama.cpp --fit would choose (llama-fit-params)."""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_server import gpu_process_memory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(r"D:\papers-tools\llama.cpp-b11136\bin")
QDIR = Path(r"D:\papers-models\q")
OUT = ROOT / "results" / "memory_placement.jsonl"
MODELS = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"]
LADDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
PORT = 8096


def fit_ngl(model):
    r = subprocess.run([str(BIN / "llama-fit-params.exe"), "-m", str(model), "-c", "2048"],
                       capture_output=True, text=True)
    m = re.search(r"-ngl (\d+)", r.stdout)
    return int(m.group(1)) if m else None


def placement(model):
    p = subprocess.Popen([str(BIN / "llama-server.exe"), "-m", str(model), "-ngl", "99", "-t", "4",
                          "-c", "2048", "-np", "1", "--port", str(PORT), "--host", "127.0.0.1", "--no-webui"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        time.sleep(2)
        return gpu_process_memory(p.pid)
    finally:
        p.kill()
        p.wait()


def main():
    for m in MODELS:
        for q in LADDER:
            path = QDIR / m / f"{m}-{q}.gguf"
            ded, sh = placement(path)
            rec = {"model": m, "quant": q, "file_gib": path.stat().st_size / 2**30,
                   "dedicated_mib": ded, "shared_mib": sh, "fit_ngl": fit_ngl(path)}
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


if __name__ == "__main__":
    main()
