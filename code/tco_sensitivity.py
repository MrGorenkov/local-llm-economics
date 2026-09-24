"""Sensitivity of the break-even volume to the electricity tariff and the GPU service life
(Qwen3.5-4B Q4_K_M on the GTX 1650, chat workload). Writes results/tco_sensitivity.json."""
import json
from pathlib import Path

import tco

ROOT = Path(__file__).resolve().parents[1]


def main():
    inp = json.loads((ROOT / "code" / "tco_inputs.json").read_text(encoding="utf-8"))
    meas = tco.load_measurements(ROOT / "results" / "energy_1650.jsonl")
    m = meas[("qwen3.5-4b/Q4_K_M", 99)]
    wl = inp["workloads"]["chat"]
    apis = {a["name"]: a for a in inp["apis"]}
    out = {"tariff": [], "lifetime": []}
    for tariff in (4.85, 8.00, 8.90):
        inp["electricity_rub_per_kwh"]["value"] = tariff
        for sname, sc in inp["scenarios"].items():
            lc = tco.local_costs(m, inp, sc, wl)
            for name in ("GigaChat Lite", "YandexGPT Pro 5.1"):
                ca = tco.api_cost(apis[name], wl)
                out["tariff"].append({"tariff": tariff, "scenario": sname, "api": name, "c_var_rub": lc["c_var_rub"],
                                      "c_fixed_rub_month": lc["c_fixed_rub_month"],
                                      "breakeven": lc["c_fixed_rub_month"] / (ca - lc["c_var_rub"])})
    inp["electricity_rub_per_kwh"]["value"] = 8.90
    for life in (24, 36, 60):
        inp["hardware"]["lifetime_months"] = life
        lc = tco.local_costs(m, inp, inp["scenarios"]["marginal_workstation"], wl)
        ca = tco.api_cost(apis["GigaChat Lite"], wl)
        out["lifetime"].append({"months": life, "c_fixed_rub_month": lc["c_fixed_rub_month"],
                                "breakeven_gigachat_lite": lc["c_fixed_rub_month"] / (ca - lc["c_var_rub"])})
    inp["hardware"]["lifetime_months"] = 36
    out["tokenizer"] = []
    lc = tco.local_costs(m, inp, inp["scenarios"]["marginal_workstation"], wl)
    for name in ("GigaChat Lite", "YandexGPT Lite", "GigaChat Pro", "YandexGPT Pro 5.1"):
        for k in (0.6, 0.8, 1.0):  # provider tokens per Qwen3.5 token for the same Russian text
            ca = tco.api_cost(apis[name], wl) * k
            out["tokenizer"].append({"api": name, "k": k, "breakeven": lc["c_fixed_rub_month"] / (ca - lc["c_var_rub"])})
    out["chat_request"] = {"energy_total_j": lc["e_req_j"], "energy_gpu_j": lc["e_gpu_j"], "t_req_s": lc["t_req_s"]}
    (ROOT / "results" / "tco_sensitivity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
