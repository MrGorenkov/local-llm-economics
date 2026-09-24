"""Build evaluation inputs from downloaded public datasets.

- WMT19 ru-en validation (newstest2018): parallel Russian/English news text for
  perplexity/KLD and the fixed prefill prompt.
- Global-MMLU (CohereLabs, Apache-2.0): a stratified sample of question ids,
  identical for the Russian and English versions.
"""
import json
import random
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "data"
SEED, N_MMLU = 20260923, 600


def wmt():
    df = pd.read_parquet(D / "wmt19_ru-en_validation.parquet")
    pairs = [(t["ru"].strip(), t["en"].strip()) for t in df["translation"]]
    print("wmt19 ru-en validation pairs:", len(pairs))
    (D / "ppl_ru.txt").write_text("\n".join(p[0] for p in pairs), encoding="utf-8")
    (D / "ppl_en.txt").write_text("\n".join(p[1] for p in pairs), encoding="utf-8")
    # the same 1000 sentence pairs in both languages, for a like-for-like KL comparison (code/kaggle_kld.py)
    (D / "ppl_ru_aligned.txt").write_text("\n".join(p[0] for p in pairs[:1000]), encoding="utf-8")
    (D / "ppl_en_aligned.txt").write_text("\n".join(p[1] for p in pairs[:1000]), encoding="utf-8")
    # prefill prompt: a block from the end of the set (energy runs only)
    (D / "pp_prompt_ru.txt").write_text(" ".join(p[0] for p in pairs[-300:]), encoding="utf-8")


def mmlu():
    ru = pd.read_parquet(D / "gmmlu_ru_test-00000-of-00001.parquet")
    en = pd.read_parquet(D / "gmmlu_en_test-00000-of-00001.parquet")
    print("Global-MMLU columns:", list(ru.columns))
    print("rows ru/en:", len(ru), len(en))
    ru = ru.set_index("sample_id")
    en = en.set_index("sample_id")
    common = ru.index.intersection(en.index)
    # culturally agnostic questions only, to separate language from culture effects
    if "cultural_sensitivity_label" in en.columns:
        ca = en.loc[common, "cultural_sensitivity_label"] == "CA"
        common = ca[ca].index
        print("culturally agnostic common ids:", len(common))
    subj = en.loc[common, "subject"]
    rng = random.Random(SEED)
    by_subj = {}
    for sid, s in subj.items():
        by_subj.setdefault(s, []).append(sid)
    for v in by_subj.values():
        v.sort()
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < N_MMLU:  # round-robin over subjects = stratified sample
        added = False
        for s in sorted(by_subj):
            if i < len(by_subj[s]) and len(picked) < N_MMLU:
                picked.append(by_subj[s][i])
                added = True
        if not added:
            break
        i += 1
    for lang, df in (("ru", ru), ("en", en)):
        with open(D / f"mmlu_{lang}_{N_MMLU}.jsonl", "w", encoding="utf-8") as f:
            for sid in picked:
                r = df.loc[sid]
                f.write(json.dumps({
                    "id": sid, "subject": r["subject"], "question": r["question"],
                    "options": [r["option_a"], r["option_b"], r["option_c"], r["option_d"]],
                    "answer": r["answer"]}, ensure_ascii=False) + "\n")
    print(f"sampled {len(picked)} ids over {len(by_subj)} subjects")

    # 5-shot exemplars: the Global-MMLU dev split (5 per subject), parallel in ru and en
    for lang in ("ru", "en"):
        dev = pd.read_parquet(D / f"gmmlu_{lang}_dev-00000-of-00001.parquet")
        shots = {}
        for _, r in dev.sort_values("sample_id").iterrows():
            shots.setdefault(r["subject"], []).append({
                "question": r["question"],
                "options": [r["option_a"], r["option_b"], r["option_c"], r["option_d"]],
                "answer": r["answer"]})
        (D / f"mmlu_{lang}_dev_shots.json").write_text(json.dumps(shots, ensure_ascii=False), encoding="utf-8")
        print(lang, "dev shots per subject:", sorted({len(v) for v in shots.values()}))


if __name__ == "__main__":
    wmt()
    mmlu()
