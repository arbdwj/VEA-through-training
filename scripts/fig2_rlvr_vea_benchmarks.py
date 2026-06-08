"""Figure 2 — VEA rate vs RLVR checkpoint across all 5 safety benchmarks.

OLMo-3.1-32B-Think RLVR sweep (steps 50..2300). One curve per benchmark. CIs are by-prompt
cluster bootstrap (rollouts nested in prompts; ICC ~0.45 -> naive CIs would be ~2x too narrow).
Reads ../data/rlvr_fortress_vea.jsonl + ../data/rlvr_bench_vea.jsonl.
"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P
from _stats import cluster_ci
from _axes import pct_top_axis

LABELS = {"fortress": "Fortress", "jbb": "JailbreakBench", "harmfulqa": "HarmfulQA",
          "agentharm": "AgentHarm", "advbench": "AdvBench"}


def load():
    # (benchmark, step) -> prompt_id -> [0/1 vea outcomes]
    g = defaultdict(lambda: defaultdict(list))
    nrows = defaultdict(int)
    for fp, force in [(P.d("rlvr_fortress_vea.jsonl"), "fortress"),
                      (P.d("rlvr_bench_vea.jsonl"), None)]:
        for l in open(fp):
            r = json.loads(l)
            if not r.get("ok"):
                continue
            bm = force or r["benchmark"]
            k = (bm, r["step"])
            g[k][r["prompt_id"]].append(int(bool(r["vea"])))
            nrows[k] += 1
    return g, nrows


def main():
    g, nrows = load()
    order = ["fortress", "jbb", "advbench", "harmfulqa", "agentharm"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for bm in order:
        steps = sorted(s for (b, s) in g if b == bm)
        if not steps:
            continue
        xs, ys, lo, hi = [], [], [], []
        for s in steps:
            rate, el, eh = cluster_ci(g[(bm, s)])
            xs.append(s); ys.append(rate); lo.append(el); hi.append(eh)
        nmed = int(sorted(nrows[(bm, s)] for s in steps)[len(steps) // 2])
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, lw=2,
                    label=f"{LABELS.get(bm, bm)} (n≈{nmed}/ckpt)")
    ax.set_xlabel("RLVR training step (checkpoint)")
    ax.set_ylabel("Verbalized eval-awareness rate (% of rollouts)")
    ax.set_title("VEA rate vs. checkpoint across safety benchmarks\n"
                 "OLMo-3.1-32B-Think (GPT-5-mini judge, Goodfire F1 · by-prompt bootstrap CIs)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    pct_top_axis(ax, 2300, label="% of final RLVR step (2300)")
    fig.tight_layout()
    fig.savefig(P.f("fig2_rlvr_vea_benchmarks.png"), dpi=140)
    print("saved ->", P.f("fig2_rlvr_vea_benchmarks.png"))


if __name__ == "__main__":
    main()
