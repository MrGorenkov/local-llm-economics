"""Produce the quantization ladder from one BF16 GGUF with llama-quantize (no imatrix).

Usage: python quantize_all.py D:\\papers-models\\src\\Qwen_Qwen3.5-0.8B-bf16.gguf qwen3.5-0.8b
"""
import subprocess
import sys
import time
from pathlib import Path

QUANTIZE = Path(r"D:\papers-tools\llama.cpp-b11136\bin\llama-quantize.exe")
OUT_ROOT = Path(r"D:\papers-models\q")
LADDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


def main(src: str, name: str):
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for q in LADDER:
        dst = out_dir / f"{name}-{q}.gguf"
        if dst.exists():
            print(f"skip {dst.name}")
            continue
        t = time.time()
        p = subprocess.run([str(QUANTIZE), src, str(dst), q], capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stderr[-1500:])
            raise SystemExit(f"quantize {q} failed")
        print(f"{dst.name}: {dst.stat().st_size / 2**30:.2f} GiB in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
