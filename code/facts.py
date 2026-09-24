"""Derived numbers quoted in the manuscript, computed from results/ (no hand arithmetic)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def csv(name):
    lines = (R / name).read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    return [dict(zip(head, l.split(","))) for l in lines[1:]]


def main():
    s = {(r["model"], r["quant"]): r for r in csv("summary_1650.csv")}
    f = lambda m, q, k: float(s[(m, q)][k])
    out = {}
    for m in ("qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"):
        out[m] = {
            "decode_power_w_Q4": f(m, "Q4_K_M", "tg_j_per_tok") * f(m, "Q4_K_M", "tg_tps"),
            "prefill_power_w_Q4": f(m, "Q4_K_M", "pp_j_per_tok") * f(m, "Q4_K_M", "pp_tps"),
            "decode_over_prefill_Q4": f(m, "Q4_K_M", "tg_j_per_tok") / f(m, "Q4_K_M", "pp_j_per_tok"),
            "Q3_vs_Q4_decode_pct": 100 * (f(m, "Q3_K_M", "tg_j_per_tok") / f(m, "Q4_K_M", "tg_j_per_tok") - 1),
            "Q2_vs_Q4_decode_pct": 100 * (f(m, "Q2_K", "tg_j_per_tok") / f(m, "Q4_K_M", "tg_j_per_tok") - 1),
            "Q8_vs_Q4_decode_pct": 100 * (f(m, "Q8_0", "tg_j_per_tok") / f(m, "Q4_K_M", "tg_j_per_tok") - 1),
            "F16_vs_Q4_decode_x": f(m, "F16", "tg_j_per_tok") / f(m, "Q4_K_M", "tg_j_per_tok"),
        }
    out["4b_Q6_vs_Q5"] = {"speed_pct": 100 * (f("qwen3.5-4b", "Q6_K", "tg_tps") / f("qwen3.5-4b", "Q5_K_M", "tg_tps") - 1),
                          "energy_pct": 100 * (f("qwen3.5-4b", "Q6_K", "tg_j_per_tok") / f("qwen3.5-4b", "Q5_K_M", "tg_j_per_tok") - 1)}
    ratio = [float(r["trace_counter_ratio"]) for r in s.values()]
    out["trace_counter_ratio_1650"] = {"min": min(ratio), "max": max(ratio), "mean": sum(ratio) / len(ratio)}

    # T4 vs 1650
    t4 = [json.loads(l) for l in (R / "energy_t4.jsonl").read_text().splitlines() if '"ok"' in l]
    agg = {}
    for r in t4:
        agg.setdefault((r["tag"], r["phase"]), []).append(r)
    med = lambda xs: sorted(xs)[len(xs) // 2]
    t4s = {k: {"j": med([x["j_per_token"] for x in v]),
               "tps": med([x["prompt_tps"] if k[1] == "pp" else x["predicted_tps"] for x in v]),
               "ratio": sum(x["energy_trace_j"] for x in v) / sum(x["energy_j"] for x in v),
               "idle": med([x["idle_loaded_w"] for x in v])} for k, v in agg.items()}
    out["t4"] = {}
    for m in ("qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"):
        for q in ("Q4_K_M", "Q8_0"):
            a, b = t4s[(f"{m}/{q}", "tg")], t4s[(f"{m}/{q}", "pp")]
            out["t4"][f"{m}/{q}"] = {
                "decode_j": a["j"], "decode_tps": a["tps"], "prefill_j": b["j"], "prefill_tps": b["tps"],
                "decode_energy_saving_vs_1650_pct": 100 * (1 - a["j"] / f(m, q, "tg_j_per_tok")),
                "prefill_energy_ratio_1650_over_t4": f(m, q, "pp_j_per_tok") / b["j"],
                "decode_speed_ratio_t4_over_1650": a["tps"] / f(m, q, "tg_tps")}
    rr = [v["ratio"] for v in t4s.values()]
    out["trace_counter_ratio_t4"] = {"min": min(rr), "max": max(rr), "mean": sum(rr) / len(rr)}
    out["t4_idle_w"] = med([v["idle"] for v in t4s.values()])

    # quality: KLD ratio ru/en
    out["kld_ru_over_en"] = {}
    for m in ("qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"):
        q = [json.loads(l) for l in (R / f"quality_{m}.jsonl").read_text(encoding="utf-8").splitlines()]
        k = {(r["quant"], r["lang"]): r for r in q if r["metric"] == "ppl_kld"}
        p = {(r["quant"], r["lang"]): r for r in q if r["metric"] in ("ppl", "ppl_kld")}
        out["kld_ru_over_en"][m] = {qq: k[(qq, "ru")]["mean_kld"] / k[(qq, "en")]["mean_kld"]
                                    for qq in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")}
        out.setdefault("ppl_increase_pct", {})[m] = {
            f"{qq}_{lang}": 100 * (p[(qq, lang)]["ppl"] / p[("F16", lang)]["ppl"] - 1)
            for qq in ("Q4_K_M", "Q3_K_M", "Q2_K") for lang in ("ru", "en")}
    tok = json.loads((R / "tokenization.json").read_text())
    out["tokens_ru_over_en"] = tok["ru_en_token_ratio"]
    (R / "facts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
