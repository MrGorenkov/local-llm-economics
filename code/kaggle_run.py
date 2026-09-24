"""Kaggle job (2x Tesla T4): build llama.cpp b11136 with CUDA, rebuild the same
quantization ladder, run quality evaluation for all models in parallel on both GPUs,
then measure per-request energy/throughput on one T4.

Inputs: private Kaggle dataset `llm-econ-inputs` (code/ and data/ of this repo).
Outputs: /kaggle/working/results/*.jsonl, logs and environment info.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

CODE = next(Path("/kaggle/input").rglob("eval_quality.py")).parent
DATA = next(Path("/kaggle/input").rglob("mmlu_ru_600.jsonl")).parent
W = Path("/kaggle/working")
RES = W / "results"
TMP = Path("/tmp/job")
SRC, BUILD = TMP / "llama.cpp", TMP / "build"
BIN = BUILD / "bin"
QDIR, MSRC = TMP / "q", TMP / "src"
TAG = "b11136"
MODELS = {
    "qwen3.5-0.8b": "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/Qwen_Qwen3.5-0.8B-bf16.gguf",
    "qwen3.5-2b": "https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-bf16.gguf",
    "qwen3.5-4b": "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/main/Qwen_Qwen3.5-4B-bf16.gguf",
}
LADDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:7.0f}s] {msg}"
    print(line, flush=True)
    with open(W / "job.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, **kw):
    log("$ " + (cmd if isinstance(cmd, str) else " ".join(map(str, cmd))))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def find_libcuda():
    """The Kaggle image ships the driver as libcuda.so.1 without the unversioned symlink CMake expects."""
    for pat in ("/usr/lib/x86_64-linux-gnu/libcuda.so*", "/usr/local/cuda/lib64/stubs/libcuda.so",
                "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so", "/usr/lib64/libcuda.so*",
                "/usr/local/nvidia/lib64/libcuda.so*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def build():
    sh(["git", "clone", "--depth", "1", "--branch", TAG, "https://github.com/ggml-org/llama.cpp", str(SRC)])
    extra = []
    lib = find_libcuda()
    log(f"libcuda: {lib}")
    if lib:
        extra.append(f"-DCUDA_cuda_driver_LIBRARY={lib}")
    else:
        extra.append("-DGGML_CUDA_NO_VMM=ON")
    base = ["cmake", "-S", str(SRC), "-B", str(BUILD), "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75",
            "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_OPENSSL=OFF"]
    try:
        sh(base + extra)
    except subprocess.CalledProcessError:
        log("cmake failed with driver library path; retrying with GGML_CUDA_NO_VMM=ON")
        shutil.rmtree(BUILD, ignore_errors=True)
        sh(base + ["-DGGML_CUDA_NO_VMM=ON"])
    sh(["cmake", "--build", str(BUILD), "-j", "4", "--target",
        "llama-server", "llama-perplexity", "llama-quantize", "llama-bench"])


def download():
    MSRC.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        dst = MSRC / f"{name}-bf16.gguf"
        sh(["curl", "-sL", "--retry", "5", "-o", str(dst), url])
        log(f"downloaded {dst.name}: {dst.stat().st_size / 2**30:.2f} GiB")


def quantize(name):
    out = QDIR / name
    out.mkdir(parents=True, exist_ok=True)
    for q in LADDER:
        sh([str(BIN / "llama-quantize"), str(MSRC / f"{name}-bf16.gguf"), str(out / f"{name}-{q}.gguf"), q],
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"quantized {name}: " + ", ".join(f"{p.name} {p.stat().st_size / 2**30:.2f}G" for p in sorted(out.glob('*.gguf'))))


def evaluate(names, gpu, port):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    for name in names:
        with open(W / f"eval_{name}.log", "w") as lf:
            subprocess.run([sys.executable, str(CODE / "eval_quality.py"), "--bin", str(BIN), "--qdir", str(QDIR),
                            "--data", str(DATA), "--model", name, "--out", str(RES / f"quality_{name}.jsonl"),
                            "--port", str(port)], env=env, stdout=lf, stderr=subprocess.STDOUT)
        log(f"GPU{gpu}: evaluation of {name} finished")


def energy(gpu=0):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    for name in MODELS:
        for q in LADDER:
            with open(W / "energy_t4.log", "a") as lf:
                subprocess.run([sys.executable, str(CODE / "bench_server.py"), "--model",
                                str(QDIR / name / f"{name}-{q}.gguf"), "--ngl", "99", "--tag", f"{name}/{q}",
                                "--server", str(BIN / "llama-server"), "--prompt-file", str(DATA / "pp_prompt_ru.txt"),
                                "--log-dir", str(W / "server_logs"), "--gpu-index", str(gpu), "--threads", "4",
                                "--out", str(RES / "energy_t4.jsonl")], env=env, stdout=lf, stderr=subprocess.STDOUT)
    log("T4 energy benchmark finished")


def env_info():
    info = {"nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout,
            "nvcc": subprocess.run("nvcc --version", shell=True, capture_output=True, text=True).stdout,
            "cpu": subprocess.run("lscpu", shell=True, capture_output=True, text=True).stdout,
            "llama_commit": subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"],
                                           capture_output=True, text=True).stdout.strip()}
    (W / "env_info.json").write_text(json.dumps(info, indent=1))


def main():
    RES.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py", "psutil"])
    dl = threading.Thread(target=download)
    dl.start()
    build()
    env_info()
    dl.join()
    for name in MODELS:
        quantize(name)
        (MSRC / f"{name}-bf16.gguf").unlink()
    g0 = threading.Thread(target=evaluate, args=(["qwen3.5-4b"], 0, 8090))
    g1 = threading.Thread(target=evaluate, args=(["qwen3.5-0.8b", "qwen3.5-2b"], 1, 8091))
    g0.start(); g1.start()
    g0.join(); g1.join()
    energy(gpu=0)
    log("done")


if __name__ == "__main__":
    main()
