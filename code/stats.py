"""Paired statistics for Global-MMLU: each quantized variant vs F16 on the same 600 questions.

- exact McNemar test (two-sided binomial on discordant pairs)
- 95 % bootstrap CI of the accuracy difference (paired resampling of questions)
- Russian-English gap per variant with a paired bootstrap CI
Writes results/mmlu_stats.csv.
"""
import json
import random
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results" / "kaggle_t4" / "results"
MODELS = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"]
LADDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
B = 10000
rng = random.Random(20260923)


def correct(model, quant, lang):
    rows = json.loads((P / f"mmlu_preds_{model}_{quant}_{lang}.json").read_text())
    return {qid: int(pred == ans) for qid, pred, ans in rows}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def boot_diff(x, y):
    """95 % CI of mean(x) - mean(y) for paired 0/1 vectors."""
    n = len(x)
    d = sorted(sum(x[i] - y[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(B))
    return d[int(0.025 * B)], d[int(0.975 * B)]


def main():
    lines = ["model,lang,quant,acc,delta_vs_f16_pp,ci_lo_pp,ci_hi_pp,mcnemar_b,mcnemar_c,p_mcnemar"]
    gap = ["model,quant,acc_ru,acc_en,gap_en_minus_ru_pp,ci_lo_pp,ci_hi_pp"]
    for m in MODELS:
        for lang in ("ru", "en"):
            ref = correct(m, "F16", lang)
            ids = sorted(ref)
            for q in LADDER:
                cur = correct(m, q, lang)
                x, y = [cur[i] for i in ids], [ref[i] for i in ids]
                b = sum(1 for i in ids if ref[i] == 1 and cur[i] == 0)
                c = sum(1 for i in ids if ref[i] == 0 and cur[i] == 1)
                lo, hi = boot_diff(x, y) if q != "F16" else (0.0, 0.0)
                lines.append(f"{m},{lang},{q},{sum(x) / len(x):.4f},{100 * (sum(x) - sum(y)) / len(x):.1f},"
                             f"{100 * lo:.1f},{100 * hi:.1f},{b},{c},{mcnemar_exact(b, c):.4g}")
        for q in LADDER:
            ru, en = correct(m, q, "ru"), correct(m, q, "en")
            ids = sorted(ru)
            x, y = [en[i] for i in ids], [ru[i] for i in ids]
            lo, hi = boot_diff(x, y)
            gap.append(f"{m},{q},{sum(y) / len(y):.4f},{sum(x) / len(x):.4f},{100 * (sum(x) - sum(y)) / len(x):.1f},"
                       f"{100 * lo:.1f},{100 * hi:.1f}")
    # Holm step-down correction over the 36 quantized-vs-F16 tests
    rows = [l.split(",") for l in lines[1:]]
    idx = [i for i, r in enumerate(rows) if r[2] != "F16"]
    order = sorted(idx, key=lambda i: float(rows[i][9]))
    m, running = len(order), 0.0
    holm = {}
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(rows[i][9])))
        holm[i] = running
    lines = [lines[0] + ",p_holm"] + [",".join(r) + f",{holm.get(i, 1.0):.4g}" for i, r in enumerate(rows)]
    (ROOT / "results" / "mmlu_stats.csv").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "results" / "mmlu_gap.csv").write_text("\n".join(gap), encoding="utf-8")
    print("\n".join(lines))
    print()
    print("\n".join(gap))


if __name__ == "__main__":
    main()
