"""Total cost of ownership (TCO) of local inference vs a pay-per-token cloud API, in rubles.

Local cost per month for V requests (each with P prompt and G generated tokens):
    C_local(V) = C_fixed + V * c_var
    C_fixed = GPU price / (lifetime months) + P_standby * H_on * tariff      (hardware + standby)
    c_var   = [(P * e_pp + G * e_tg) * k_psu + P_platform * t_req] / 3.6e6 * tariff
where e_pp, e_tg are measured GPU joules per token, t_req = P / v_pp + G / v_tg is the
request latency, P_platform the rest-of-system power while busy (scenario parameter),
k_psu = 1 / PSU efficiency.
Cloud cost: C_api(V) = V * (P * p_in + G * p_out). Break-even V* = C_fixed / (c_api - c_var).
Capacity: V_max = H_on * 3600 / t_req requests per month.

All inputs live in tco_inputs.json (prices with source URLs and access dates).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_measurements(path):
    """Median GPU J/token and tokens/s per (tag, phase) from energy_*.jsonl."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if '"status": "ok"' in l]
    agg = {}
    for r in rows:
        key = (r["tag"], r["ngl"], r["phase"])
        agg.setdefault(key, []).append(r)
    out = {}
    for (tag, ngl, phase), rs in agg.items():
        jpt = sorted(x["j_per_token"] for x in rs)
        tps = sorted((x["prompt_tps"] if phase == "pp" else x["predicted_tps"]) for x in rs)
        out.setdefault((tag, ngl), {})[phase] = {"j_per_token": jpt[len(jpt) // 2],
                                                "tps": tps[len(tps) // 2], "n": len(rs)}
    return out


def local_costs(m, inp, scenario, wl):
    P, G = wl["prompt_tokens"], wl["gen_tokens"]
    tariff = inp["electricity_rub_per_kwh"]["value"]
    k_psu = 1.0 / scenario["psu_efficiency"]
    t_req = P / m["pp"]["tps"] + G / m["tg"]["tps"]
    e_gpu = P * m["pp"]["j_per_token"] + G * m["tg"]["j_per_token"]
    # standby power is charged in the fixed cost for every hour the machine is on, so the variable part
    # counts only the energy above standby while a request runs (no double counting)
    e_req = (e_gpu + scenario["platform_active_w"] * t_req) * k_psu
    e_net = e_req - scenario["standby_w"] * k_psu * t_req
    c_var = e_net / 3.6e6 * tariff
    hw = inp["hardware"]
    c_fixed = (hw["gpu_price_rub"]["value"] / hw["lifetime_months"]
               + scenario["standby_w"] * k_psu * scenario["hours_on_per_month"] / 1000 * tariff)
    v_max = scenario["hours_on_per_month"] * 3600 / t_req
    return {"t_req_s": t_req, "e_req_j": e_req, "e_gpu_j": e_gpu, "c_var_rub": c_var,
            "c_fixed_rub_month": c_fixed, "v_max_month": v_max}


def api_cost(api, wl):
    return wl["prompt_tokens"] * api["rub_per_1k_in"] / 1000 + wl["gen_tokens"] * api["rub_per_1k_out"] / 1000


def main():
    inp = json.loads((ROOT / "code" / "tco_inputs.json").read_text(encoding="utf-8"))
    meas = load_measurements(ROOT / "results" / "energy_1650.jsonl")
    rows = []
    for (tag, ngl), m in sorted(meas.items()):
        if "pp" not in m or "tg" not in m or tag.startswith(("cliff:", "cpu:")):
            continue
        for wname, wl in inp["workloads"].items():
            for sname, sc in inp["scenarios"].items():
                lc = local_costs(m, inp, sc, wl)
                for api in inp["apis"]:
                    ca = api_cost(api, wl)
                    be = lc["c_fixed_rub_month"] / (ca - lc["c_var_rub"]) if ca > lc["c_var_rub"] else None
                    rows.append({"tag": tag, "ngl": ngl, "workload": wname, "scenario": sname,
                                 "api": api["name"], **lc, "c_api_rub": ca, "breakeven_req_month": be,
                                 "breakeven_within_capacity": (be is not None and be <= lc["v_max_month"]),
                                 "local_rub_per_1m_tokens_var": lc["c_var_rub"] /
                                 (wl["prompt_tokens"] + wl["gen_tokens"]) * 1e6})
    out = ROOT / "results" / "tco.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
