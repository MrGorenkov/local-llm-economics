"""Probe NVML capabilities on this GPU: power reading, read latency, energy counter."""
import time

import pynvml as nvml

nvml.nvmlInit()
h = nvml.nvmlDeviceGetHandleByIndex(0)
print("device:", nvml.nvmlDeviceGetName(h))
print("power now, W:", nvml.nvmlDeviceGetPowerUsage(h) / 1000)
print("enforced limit, W:", nvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000)

t = time.perf_counter()
for _ in range(200):
    nvml.nvmlDeviceGetPowerUsage(h)
print("200 power reads, s:", round(time.perf_counter() - t, 3))

try:
    e0 = nvml.nvmlDeviceGetTotalEnergyConsumption(h)
    time.sleep(1.0)
    e1 = nvml.nvmlDeviceGetTotalEnergyConsumption(h)
    print("energy counter available; mJ over 1 s:", e1 - e0)
except nvml.NVMLError as e:
    print("energy counter unavailable:", e)

# how often does the reported power value actually change?
vals, ts = [], []
t0 = time.perf_counter()
while time.perf_counter() - t0 < 2.0:
    vals.append(nvml.nvmlDeviceGetPowerUsage(h))
    ts.append(time.perf_counter())
changes = [ts[i] for i in range(1, len(vals)) if vals[i] != vals[i - 1]]
if len(changes) > 1:
    gaps = [b - a for a, b in zip(changes, changes[1:])]
    print("samples:", len(vals), "value changes:", len(changes),
          "median update interval, ms:", round(1000 * sorted(gaps)[len(gaps) // 2], 1))
else:
    print("samples:", len(vals), "value changes:", len(changes))
nvml.nvmlShutdown()
