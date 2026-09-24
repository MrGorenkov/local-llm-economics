# Energy, cost and Russian-language quality of quantized small language models on an entry-level GPU

Code, measurement logs and figures for the article
"Energy, cost and Russian-language quality of quantized small language models on an entry-level GPU"
(A. A. Gorenkov, Plekhanov Russian University of Economics, 2026).

## What is measured
- **Energy and speed** of Qwen3.5 0.8B / 2B / 4B in seven GGUF formats (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K)
  on an NVIDIA GeForce GTX 1650 4 GB (Windows 10, llama.cpp b11136) and on a Tesla T4 (Kaggle, Linux).
  Energy per request is read from the NVML hardware energy counter (`nvmlDeviceGetTotalEnergyConsumption`)
  around each window of back-to-back requests (≥ 5 s); prefill and decode are measured separately.
- **Memory placement** under Windows (dedicated vs shared GPU memory) and three offloading strategies.
- **Quality**: perplexity and KL divergence from F16 on WMT newstest2018 (parallel ru/en), five-shot Global-MMLU
  accuracy on the same 600 questions in Russian and English, McNemar tests and bootstrap intervals.
- **Cost of ownership** in rubles against GigaChat, YandexGPT and gpt-5-nano, with the 2026 Moscow tariff.

## Layout
```
code/
  quantize_all.py       BF16 GGUF -> 7 formats with llama-quantize (no imatrix)
  prepare_data.py       WMT19 ru-en validation -> ppl_{ru,en}.txt; Global-MMLU sample (600 ids) + 5-shot exemplars
  bench_server.py       per-request energy/throughput on a running llama-server (NVML counter + 20 Hz trace)
  run_local.py          GTX 1650 campaign: ladder, VRAM-cliff strategies, CPU-only runs
  memory_placement.py   dedicated/shared GPU memory per variant, llama.cpp --fit decision
  eval_quality.py       perplexity, KL divergence (llama-perplexity), Global-MMLU 5-shot (llama-server)
  kaggle_run.py         the Kaggle T4 job: build llama.cpp b11136, quantize, evaluate, measure energy
  kaggle_kld.py         Kaggle job: KL divergence on sentence-aligned ru/en text (first 1000 pairs)
  stats.py              McNemar tests, bootstrap CIs, Russian-English gap
  tco.py, tco_inputs.json, tco_sensitivity.py   cost model, prices with sources, sensitivity
  analyze.py, facts.py  tables, figures and every number quoted in the text
data/                   not in the repository: rebuilt from public datasets by code/prepare_data.py
results/                raw measurement logs (*.jsonl), summaries (*.csv, *.json)
paper/figures/          figures of the article
```

## Reproduce
1. Windows, NVIDIA driver ≥ 546, Python 3.12: `pip install nvidia-ml-py psutil numpy pandas pyarrow matplotlib gguf tokenizers`.
2. Download llama.cpp b11136 (`llama-b11136-bin-win-cuda-12.4-x64.zip` + cudart) and the BF16 GGUF files of
   Qwen3.5-0.8B/2B/4B (bartowski conversions on Hugging Face); adjust the paths at the top of the scripts.
3. `python code/quantize_all.py <bf16.gguf> <name>` for each model; `python code/prepare_data.py`.
4. `python code/run_local.py ladder`, then `cliff` and `cpu`; `python code/memory_placement.py`.
5. Quality on any CUDA GPU: `python code/eval_quality.py ...` or the Kaggle job `code/kaggle_run.py`.
6. `python code/stats.py`, `python code/tco.py`, `python code/analyze.py`, `python code/facts.py`.

## Data and licenses
- Qwen3.5 models: Apache-2.0.
- Global-MMLU (CohereLabs): Apache-2.0. WMT newstest2018: WMT shared-task test set, research use.
- Prices and tariffs: public web pages accessed on 23.09.2026 (URLs in `code/tco_inputs.json`).
- Code in this repository: MIT.
