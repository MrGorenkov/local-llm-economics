"""Tables and figures for the article from results/*.jsonl.

Figures are sized for a 165 mm text width (NTV ITMO template), exported as PNG (600 dpi)
and PDF. Colour follows the model size (validated categorical slots 1-3); every series
also has its own marker and line style so the figures survive greyscale printing.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MODELS = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"]
LABEL = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-2b": "Qwen3.5-2B", "qwen3.5-4b": "Qwen3.5-4B"}
LADDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
COLOR = {"qwen3.5-0.8b": "#2a78d6", "qwen3.5-2b": "#eb6834", "qwen3.5-4b": "#1baf7a"}
MARKER = {"qwen3.5-0.8b": "o", "qwen3.5-2b": "s", "qwen3.5-4b": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.5, "lines.markersize": 5,
})


def jl(name):
    p = RES / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def med(xs):
    return st.median(xs) if xs else None


def energy_table(rows):
    """Median per (tag, ngl, phase) with interquartile range."""
    out = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        out.setdefault((r["tag"], r["ngl"], r["phase"]), []).append(r)
    tab = {}
    for k, rs in out.items():
        j = sorted(x["j_per_token"] for x in rs)
        t = [x["prompt_tps"] if k[2] == "pp" else x["predicted_tps"] for x in rs]
        q1, q3 = (st.quantiles(j, n=4)[0], st.quantiles(j, n=4)[2]) if len(j) >= 4 else (j[0], j[-1])
        tab[k] = {"j": med(j), "j_q1": q1, "j_q3": q3, "tps": med(t), "n": len(rs),
                  "cpu": med([x["mean_cpu_util"] for x in rs]),
                  "power": med([x["energy_j"] / x["wall_s"] for x in rs]),
                  "shared_mib": med([x.get("proc_shared_mib") or 0 for x in rs]),
                  "trace_ratio": sum(x["energy_trace_j"] for x in rs) / sum(x["energy_j"] for x in rs)}
    return tab


def bpw(model, quant):
    """Effective bits per weight relative to the F16 file (16 bits)."""
    f = Path(r"D:\papers-models\q") / model
    return 16 * (f / f"{model}-{quant}.gguf").stat().st_size / (f / f"{model}-F16.gguf").stat().st_size


XPOS = {q: i for i, q in enumerate(LADDER)}


def plain_log(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


SHORT = {"F16": "F16", "Q8_0": "Q8", "Q6_K": "Q6", "Q5_K_M": "Q5", "Q4_K_M": "Q4", "Q3_K_M": "Q3", "Q2_K": "Q2"}


def quant_axis(ax, start=0):
    """Ordered weight formats; short labels (Q6/Q2 = *_K, Q5-Q3 = *_K_M, Q8 = Q8_0) explained in captions."""
    ax.set_xticks(range(start, len(LADDER)))
    ax.set_xticklabels([SHORT[q] for q in LADDER[start:]], fontsize=7)
    ax.set_xlim(start - 0.4, len(LADDER) - 0.6)


def model_language_legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[m], marker=MARKER[m], label=LABEL[m]) for m in MODELS]
    h += [Line2D([], [], color=MUTED, ls="-", label="Russian"), Line2D([], [], color=MUTED, ls="--", label="English")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=7)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_energy(tab, placement):
    fig, axes = plt.subplots(1, 2, figsize=(165 * MM, 62 * MM))
    for ax, phase, title in ((axes[0], "tg", "Decode (generation)"), (axes[1], "pp", "Prefill (prompt)")):
        for m in MODELS:
            xs, ys, spilled = [], [], []
            for q in LADDER:
                v = tab.get((f"{m}/{q}", 99, phase))
                if v:
                    xs.append(XPOS[q])
                    ys.append(v["j"])
                    spilled.append((placement.get((m, q), {}).get("shared_mib") or 0) > 150)
            ax.plot(xs, ys, color=COLOR[m], marker=MARKER[m], label=LABEL[m], zorder=2)
            for x, y, s in zip(xs, ys, spilled):
                if s:  # hollow marker: the weights overflowed into shared system memory
                    ax.plot(x, y, marker=MARKER[m], markersize=5, markerfacecolor="white",
                            markeredgecolor=COLOR[m], markeredgewidth=1.3, zorder=3)
        plain_log(ax)
        quant_axis(ax)
        ax.set_xlabel("Weight format")
        ax.set_ylabel("GPU energy, J/token")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    axes[0].legend(loc="upper right")
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig1_energy_vs_quant")


def fig_cliff(tab, fit_ngl):
    """Decode speed vs layers on GPU, one panel per variant; three placement strategies."""
    cases = [("qwen3.5-4b", "Q4_K_M"), ("qwen3.5-4b", "Q6_K"), ("qwen3.5-4b", "Q8_0"),
             ("qwen3.5-4b", "F16"), ("qwen3.5-2b", "F16")]
    fig, axes = plt.subplots(1, 5, figsize=(165 * MM, 50 * MM))
    for ax, (m, q) in zip(axes, cases):
        tag = f"cliff:{m}/{q}"
        pts = sorted((ngl, v["tps"]) for (t, ngl, ph), v in tab.items() if t == tag and ph == "tg" and ngl >= 0)
        n_layers = 33 if m.endswith("4b") else 25
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[m], marker=MARKER[m], ms=4)
        full = tab.get((f"{m}/{q}", 99, "tg"))
        if full:
            ax.plot(n_layers, full["tps"], marker="D", ms=5, markerfacecolor="white", markeredgecolor=INK, zorder=4)
        auto = tab.get((tag, -1, "tg"))
        if auto and fit_ngl.get((m, q)) is not None:
            ax.plot(fit_ngl[(m, q)], auto["tps"], marker="*", ms=9, color=INK, zorder=4)
        ax.set_title(f"{LABEL[m][8:]} {q}", fontsize=7.5, color=INK, loc="left")
        ax.set_xlim(-2, n_layers + 3)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Layers on GPU", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Decode speed, tokens/s")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=MUTED, marker="o", ms=4, label="explicit CPU offload"),
         Line2D([], [], ls="", marker="*", ms=9, color=INK, label="llama.cpp automatic fit"),
         Line2D([], [], ls="", marker="D", ms=5, markerfacecolor="white", markeredgecolor=INK,
                label="all layers requested (driver fallback)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09), fontsize=7)
    fig.subplots_adjust(wspace=0.35, bottom=0.25)
    save(fig, "fig2_vram_cliff")

def fig_quality(qrows):
    fig, axes = plt.subplots(1, 3, figsize=(165 * MM, 58 * MM))
    for m in MODELS:
        for lang, ls in (("ru", "-"), ("en", "--")):
            k = [(XPOS[r["quant"]], r) for r in qrows if r["model"] == m and r["lang"] == lang
                 and r["metric"] == "ppl_kld"]
            k.sort(key=lambda t: t[0])
            if k:
                axes[0].plot([x for x, _ in k], [r["mean_kld"] for _, r in k], ls=ls, color=COLOR[m],
                             marker=MARKER[m], label=f"{LABEL[m]} {lang.upper()}")
                axes[1].plot([x for x, _ in k], [100 - r["same_top_p"] for _, r in k], ls=ls, color=COLOR[m],
                             marker=MARKER[m])
            a = [(XPOS[r["quant"]], r) for r in qrows if r["model"] == m and r["lang"] == lang
                 and r["metric"] == "mmlu"]
            a.sort(key=lambda t: t[0])
            if a:
                axes[2].plot([x for x, _ in a], [100 * r["accuracy"] for _, r in a], ls=ls, color=COLOR[m],
                             marker=MARKER[m])
    plain_log(axes[0])
    axes[0].set_ylabel("Mean KL divergence from F16")
    axes[1].set_ylabel("Top-token mismatch vs F16, %")
    axes[2].set_ylabel("Global-MMLU 5-shot accuracy, %")
    for i, ax in enumerate(axes):
        quant_axis(ax, start=1 if i < 2 else 0)
        ax.set_xlabel("Weight format")
    for ax, t in zip(axes, ("(a)", "(b)", "(c)")):
        ax.set_title(t, fontsize=8, color=INK, loc="left")
    model_language_legend(fig)
    fig.subplots_adjust(wspace=0.42, bottom=0.2)
    save(fig, "fig3_quality")


def fig_tradeoff_cost(tab, qrows, placement=None):
    """(a) decode energy vs Russian Global-MMLU accuracy with the Pareto frontier;
    (b) monthly cost vs volume: local GTX 1650 (Qwen3.5-4B Q4_K_M, chat workload,
    workstation scenario) against per-token APIs, from results/tco.jsonl."""
    fig, axes = plt.subplots(1, 2, figsize=(165 * MM, 66 * MM))
    ax = axes[0]
    pts = []
    for m in MODELS:
        acc = {r["quant"]: 100 * r["accuracy"] for r in qrows
               if r["model"] == m and r["lang"] == "ru" and r["metric"] == "mmlu"}
        for q in LADDER:
            v = tab.get((f"{m}/{q}", 99, "tg"))
            if v and q in acc:
                pts.append((v["j"], acc[q], m, q))
        for p in (p for p in pts if p[2] == m):
            spilled = ((placement or {}).get((m, p[3]), {}).get("shared_mib") or 0) > 150
            ax.scatter([p[0]], [p[1]], color="white" if spilled else COLOR[m], marker=MARKER[m], s=22,
                       zorder=3, edgecolors=COLOR[m], linewidths=1.0)
        ax.scatter([], [], color=COLOR[m], marker=MARKER[m], s=22, label=LABEL[m])
    front, best = [], -1
    for p in sorted(pts, key=lambda t: t[0]):
        if p[1] > best:
            front.append(p)
            best = p[1]
    ax.plot([p[0] for p in front], [p[1] for p in front], color=MUTED, lw=1, ls="--", zorder=2)
    keylab = {("qwen3.5-0.8b", "Q2_K"): (6, -2), ("qwen3.5-0.8b", "Q4_K_M"): (7, -12),
              ("qwen3.5-2b", "Q4_K_M"): (-8, 7), ("qwen3.5-4b", "Q4_K_M"): (-12, 6),
              ("qwen3.5-4b", "Q8_0"): (-10, -11), ("qwen3.5-4b", "F16"): (-14, -11)}
    short_m = {"qwen3.5-0.8b": "0.8B", "qwen3.5-2b": "2B", "qwen3.5-4b": "4B"}
    for p in pts:
        if (p[2], p[3]) in keylab:
            ax.annotate(f"{short_m[p[2]]} {SHORT[p[3]]}", (p[0], p[1]), textcoords="offset points",
                        xytext=keylab[(p[2], p[3])], fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Decode energy on GTX 1650, J/token (log)")
    ax.set_ylabel("Global-MMLU (Russian), %")
    ax.set_title("(a)", fontsize=8, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[1]
    rows = [r for r in jl("tco.jsonl") if r["tag"] == "qwen3.5-4b/Q4_K_M" and r["workload"] == "chat"
            and r["scenario"] == "marginal_workstation"]
    import numpy as np
    v = np.logspace(2, np.log10(3e5), 200)
    r0 = rows[0]
    cap = r0["v_max_month"]
    vl = v[v <= cap]
    ax.plot(vl, r0["c_fixed_rub_month"] + r0["c_var_rub"] * vl, color=COLOR["qwen3.5-4b"], lw=2,
            label="Local GTX 1650, Qwen3.5-4B Q4_K_M")
    ax.axvline(cap, color=COLOR["qwen3.5-4b"], lw=0.8, ls=":")
    styles = {"GigaChat Lite": ":", "YandexGPT Lite": "-.", "GigaChat Pro": "--", "YandexGPT Pro 5.1": "-"}
    for r in rows:
        if r["api"] not in styles:
            continue
        ax.plot(v, r["c_api_rub"] * v, color=INK, lw=1, ls=styles[r["api"]])
        ax.annotate(r["api"], (v[-1], r["c_api_rub"] * v[-1]), textcoords="offset points", xytext=(3, 0),
                    fontsize=6.5, color=INK, va="center")
        be = r["breakeven_req_month"]
        if be and be <= cap:
            ax.plot(be, r["c_api_rub"] * be, marker="o", ms=4, color=INK, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " ")))
    ax.set_xlabel("Requests per month (300 + 300 tokens)")
    ax.set_ylabel("Monthly cost, RUB")
    ax.set_title("(b)", fontsize=8, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig4_tradeoff_cost")


def main():
    erows = jl("energy_1650.jsonl")
    tab = energy_table(erows)
    placement = {(r["model"], r["quant"]): r for r in jl("memory_placement.jsonl")}
    fit_ngl = {k: v.get("fit_ngl") for k, v in placement.items()}
    fig_energy(tab, placement)
    fig_cliff(tab, fit_ngl)
    # fidelity (perplexity, KL) from the sentence-aligned ru/en text, accuracy from the main quality run
    qrows = []
    for m in MODELS:
        qrows += [r for r in jl(f"quality_{m}.jsonl") if r["metric"] == "mmlu"]
        qrows += jl(f"quality_aligned_{m}.jsonl")
    if qrows:
        fig_quality(qrows)
        if (RES / "tco.jsonl").exists():
            fig_tradeoff_cost(tab, qrows, placement)
    # summary table
    lines = ["model,quant,bpw,file_gib,dedicated_mib,shared_mib,pp_tps,pp_j_per_tok,tg_tps,tg_j_per_tok,"
             "tg_j_q1,tg_j_q3,gpu_power_tg_w,cpu_util_tg,trace_counter_ratio"]
    for m in MODELS:
        for q in LADDER:
            pp, tg = tab.get((f"{m}/{q}", 99, "pp")), tab.get((f"{m}/{q}", 99, "tg"))
            pl = placement.get((m, q), {})
            if pp and tg:
                lines.append(f"{m},{q},{bpw(m, q):.2f},{pl.get('file_gib', 0):.2f},{pl.get('dedicated_mib') or 0:.0f},"
                             f"{pl.get('shared_mib') or 0:.0f},{pp['tps']:.1f},{pp['j']:.4f},{tg['tps']:.1f},{tg['j']:.4f},"
                             f"{tg['j_q1']:.4f},{tg['j_q3']:.4f},{tg['power']:.1f},{tg['cpu']:.0f},{tg['trace_ratio']:.3f}")
    (RES / "summary_1650.csv").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

