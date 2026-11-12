"""Figure 7 — First-VEA-token by benchmark: blunt prompts trigger eval-awareness earlier.

CDF of the first-VEA-token index per benchmark. Reads data/rlvr/vea_first_token.csv.
"""
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P

LABELS = {"jbb": "JailbreakBench", "advbench": "AdvBench", "agentharm": "AgentHarm",
          "harmfulqa": "HarmfulQA", "fortress": "Fortress"}
COL = {"jbb": "tab:orange", "advbench": "tab:green", "agentharm": "tab:purple",
       "harmfulqa": "tab:red", "fortress": "tab:blue"}
ORDER = ["jbb", "advbench", "agentharm", "harmfulqa", "fortress"]

by = defaultdict(list)
with open(P.d("rlvr", "vea_first_token.csv")) as f:
    for row in csv.DictReader(f):
        by[row["benchmark"]].append(float(row["first_tok"]))

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for b in ORDER:
    v = np.sort(np.array(by[b]))
    if not len(v):
        continue
    cdf = np.arange(1, len(v) + 1) / len(v)
    med = np.median(v)
    ax.plot(v, 100 * cdf, lw=2.2, color=COL[b],
            label=f"{LABELS[b]}  (median {med:.0f} tok, n={len(v)})")
    ax.plot([med, med], [0, 50], color=COL[b], ls=":", lw=1, alpha=.6)
ax.axhline(50, color="gray", lw=.7, ls="--", alpha=.6)
ax.set_xlim(0, 700)
ax.set_xlabel("first VEA token index in CoT  (exact OLMo-3 tokenizer)")
ax.set_ylabel("cumulative % of VEA-positive rollouts")
ax.set_title("Blunt prompts trigger eval-awareness earlier than realistic ones\n"
             "first-VEA-token CDF by benchmark · OLMo-3.1-32B-Think RLVR · curves left = earlier VEA")
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(P.f("fig7_first_vea_by_benchmark.png"), dpi=140)
print("saved ->", P.f("fig7_first_vea_by_benchmark.png"))
