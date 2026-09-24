"""Kaggle job (T4): KL divergence and perplexity on sentence-aligned Russian/English text
(the first 1000 sentence pairs of newstest2018, whole files), reusing the build/quantize
steps of kaggle_run.py. GPU0 handles the 4B model, GPU1 the 0.8B and 2B models."""
import os
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(next(Path('/kaggle/input').rglob('kaggle_run.py')).parent))
import kaggle_run as K  # noqa: E402

RES = K.W / "results"


def evaluate(names, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    for name in names:
        with open(K.W / f"kld_{name}.log", "w") as lf:
            subprocess.run([sys.executable, str(K.CODE / "eval_quality.py"), "--bin", str(K.BIN), "--qdir", str(K.QDIR),
                            "--data", str(K.DATA), "--model", name, "--out", str(RES / f"quality_aligned_{name}.jsonl"),
                            "--skip-mmlu", "--chunks", "0", "--ppl-suffix", "_aligned"],
                           env=env, stdout=lf, stderr=subprocess.STDOUT)
        K.log(f"GPU{gpu}: aligned KLD of {name} finished")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    K.TMP.mkdir(parents=True, exist_ok=True)
    dl = threading.Thread(target=K.download)
    dl.start()
    K.build()
    dl.join()
    for name in K.MODELS:
        K.quantize(name)
        (K.MSRC / f"{name}-bf16.gguf").unlink()
    g0 = threading.Thread(target=evaluate, args=(["qwen3.5-4b"], 0))
    g1 = threading.Thread(target=evaluate, args=(["qwen3.5-0.8b", "qwen3.5-2b"], 1))
    g0.start(); g1.start(); g0.join(); g1.join()
    K.log("done")


if __name__ == "__main__":
    main()
