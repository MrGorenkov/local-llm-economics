"""Quality of quantized models relative to F16, on parallel Russian and English inputs.

1. Perplexity and KL divergence vs the F16 model (llama-perplexity) on WMT19 ru-en
   newstest2018: identical content in both languages.
2. Zero-shot multiple-choice accuracy on a stratified 600-question sample of
   Global-MMLU (culturally agnostic subset), same question ids in ru and en,
   scored by the next-token probabilities of the answer letters.

Runs on Windows (local) and Linux (Kaggle); binaries and folders are arguments.
Appends JSON lines to --out.

Example:
  python eval_quality.py --bin D:/papers-tools/llama.cpp-b11136/bin --qdir D:/papers-models/q \
      --data ../data --model qwen3.5-0.8b --out ../results/quality.jsonl
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LADDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
EXE = ".exe" if os.name == "nt" else ""
PORT = 8090
HEADER = {
    "ru": "Ниже приведены вопросы с четырьмя вариантами ответа. Для каждого выберите один правильный.\n\n",
    "en": "The following are multiple-choice questions with four options. Choose the correct one for each.\n\n",
}
BLOCK = {
    "ru": "Вопрос: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nОтвет:",
    "en": "Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:",
}
LETTERS = "ABCD"
N_SHOTS = 5


def build_prompt(it, lang, shots):
    """Standard MMLU 5-shot prompt: exemplars from the same subject's dev split."""
    parts = [HEADER[lang]]
    for s in shots.get(it["subject"], [])[:N_SHOTS]:
        a, b, c, d = s["options"]
        parts.append(BLOCK[lang].format(q=s["question"], a=a, b=b, c=c, d=d) + f" {s['answer']}\n\n")
    a, b, c, d = it["options"]
    parts.append(BLOCK[lang].format(q=it["question"], a=a, b=b, c=c, d=d))
    return "".join(parts)


def ppl_kld(bin_dir, model_path, base_path, text, chunks, ngl, make_base):
    args = [str(bin_dir / f"llama-perplexity{EXE}"), "-m", str(model_path), "-f", str(text),
            "-c", "512", "--chunks", str(chunks if chunks > 0 else -1), "-ngl", str(ngl), "--kl-divergence-base", str(base_path)]
    if not make_base:
        args.append("--kl-divergence")
    t = time.time()
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    if p.returncode != 0:
        raise RuntimeError(out[-2000:])
    res = {"seconds": round(time.time() - t, 1)}
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", out)
    if m:
        res["ppl"], res["ppl_se"] = float(m.group(1)), float(m.group(2))
    patterns = {
        "mean_kld": r"Mean\s+KLD:\s+([-\d.]+)\s+±\s+([\d.]+)",
        "same_top_p": r"Same top p:\s+([\d.]+)\s+±\s+([\d.]+)\s*%",
        "mean_ln_ppl_ratio": r"Mean\s+ln\(PPL\(Q\)/PPL\(base\)\):\s+([-\d.]+)\s+±\s+([\d.]+)",
        "kld_99": r"99\.0%\s+KLD:\s+([-\d.]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, out)
        if m:
            res[k] = float(m.group(1))
            if m.lastindex and m.lastindex >= 2:
                res[k + "_se"] = float(m.group(2))
    if "ppl" not in res:  # KLD mode prints PPL(Q) instead of "Final estimate"
        m = re.search(r"Mean PPL\(Q\)\s*:\s*([\d.]+)\s+±\s+([\d.]+)", out)
        if m:
            res["ppl"], res["ppl_se"] = float(m.group(1)), float(m.group(2))
    return res


def server(bin_dir, model_path, ngl, ctx=2048):
    args = [str(bin_dir / f"llama-server{EXE}"), "-m", str(model_path), "-ngl", str(ngl),
            "-c", str(ctx), "-np", "1", "--port", str(PORT), "--host", "127.0.0.1", "--no-webui"]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError("server exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server not ready")


def completion(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def mmlu(items, lang, shots):
    correct, no_letter, rows = 0, 0, []
    for it in items:
        prompt = build_prompt(it, lang, shots)
        res = completion({"prompt": prompt, "n_predict": 1, "n_probs": 20, "temperature": 0.0,
                          "cache_prompt": False})
        probs = res["completion_probabilities"][0]
        cands = probs.get("top_logprobs") or probs.get("top_probs") or probs.get("probs") or []
        score = {L: 0.0 for L in LETTERS}
        for cnd in cands:
            tok = (cnd.get("token") or cnd.get("tok_str") or "").strip()
            p = cnd.get("prob")
            if p is None and "logprob" in cnd:
                p = math.exp(cnd["logprob"])
            if tok in score:
                score[tok] += p or 0.0
        pred = max(score, key=score.get) if any(score.values()) else None
        no_letter += pred is None
        ok = pred == it["answer"]
        correct += ok
        rows.append((it["id"], pred, it["answer"]))
    return {"n": len(items), "accuracy": correct / len(items), "no_letter": no_letter}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="folder/prefix name, e.g. qwen3.5-2b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=24)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--quants", default=",".join(LADDER))
    ap.add_argument("--skip-mmlu", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--mmlu-n", type=int, default=600, help="use the first N questions (smoke tests)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--ppl-suffix", default="", help="e.g. _aligned: identical sentences in both languages")
    a = ap.parse_args()
    globals()["PORT"] = a.port
    bin_dir, qdir, data = Path(a.bin), Path(a.qdir) / a.model, Path(a.data)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quants = a.quants.split(",")

    def write(rec):
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    if not a.skip_ppl:
        for lang in ("ru", "en"):
            base = qdir / f"kld_base_{lang}.bin"
            text = data / f"ppl_{lang}{a.ppl_suffix}.txt"
            f16 = qdir / f"{a.model}-F16.gguf"
            r = ppl_kld(bin_dir, f16, base, text, a.chunks, a.ngl, make_base=True)
            write({"model": a.model, "quant": "F16", "lang": lang, "metric": "ppl", **r})
            for q in quants:
                if q == "F16":
                    continue
                r = ppl_kld(bin_dir, qdir / f"{a.model}-{q}.gguf", base, text, a.chunks, a.ngl, make_base=False)
                write({"model": a.model, "quant": q, "lang": lang, "metric": "ppl_kld", **r})
            base.unlink(missing_ok=True)

    if not a.skip_mmlu:
        items = {lang: [json.loads(l) for l in open(data / f"mmlu_{lang}_600.jsonl", encoding="utf-8")][:a.mmlu_n]
                 for lang in ("ru", "en")}
        shots = {lang: json.loads((data / f"mmlu_{lang}_dev_shots.json").read_text(encoding="utf-8"))
                 for lang in ("ru", "en")}
        for q in quants:
            proc = server(bin_dir, qdir / f"{a.model}-{q}.gguf", a.ngl, ctx=4096)
            try:
                for lang in ("ru", "en"):
                    t = time.time()
                    summary, rows = mmlu(items[lang], lang, shots[lang])
                    write({"model": a.model, "quant": q, "lang": lang, "metric": "mmlu",
                           "seconds": round(time.time() - t, 1), **summary})
                    with open(out.with_name(f"mmlu_preds_{a.model}_{q}_{lang}.json"), "w", encoding="utf-8") as f:
                        json.dump(rows, f)
            finally:
                proc.terminate()
                proc.wait(timeout=30)


if __name__ == "__main__":
    sys.exit(main())
