"""Figure 2 — VEA rate vs RLVR checkpoint across all 5 safety benchmarks.

OLMo-3.1-32B-Think RLVR sweep (steps 50..2300). One curve per benchmark. CIs are by-prompt
cluster bootstrap (rollouts nested in prompts; ICC ~0.45 -> naive CIs would be ~2x too narrow).
Reads data/rlvr/vea.jsonl.
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
    for r in P.rows("vea.jsonl", "rlvr"):
        if not r.get("ok"):
            continue
        k = (r["benchmark"], r["step"])
        g[k][r["prompt_id"]].append(int(bool(r["vea"])))
        nrows[k] += 1
    return g, nrows


def main():
    g, nrows = load()
    order = ["fortress", "jbb", "advbench", "harmfulqa", "agentharm"]
    fig, ax = plt.subplots(figsize=(10, 6))
    by_step = defaultdict(list)   # step -> list of per-benchmark rates (for the average line)
    for bm in order:
        steps = sorted(s for (b, s) in g if b == bm)
        if not steps:
            continue
        xs, ys, lo, hi = [], [], [], []
        for s in steps:
            rate, el, eh = cluster_ci(g[(bm, s)])
            xs.append(s); ys.append(rate); lo.append(el); hi.append(eh)
            by_step[s].append(rate)
        nmed = int(sorted(nrows[(bm, s)] for s in steps)[len(steps) // 2])
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, lw=1.6, alpha=0.7,
                    label=f"{LABELS.get(bm, bm)} (n≈{nmed}/ckpt)")

    # bold average-over-benchmarks line (steps where all 5 benchmarks are present)
    avg_steps = sorted(s for s in by_step if len(by_step[s]) == len(order))
    avg = [sum(by_step[s]) / len(by_step[s]) for s in avg_steps]
    ax.plot(avg_steps, avg, color="black", marker="o", ms=9, lw=3.4, ls="--", zorder=10,
            label="Average (mean over 5 benchmarks)")
    if len(avg) >= 2:
        ax.annotate(f"{avg[0]:.0f}%", (avg_steps[0], avg[0]), textcoords="offset points",
                    xytext=(-2, -16), ha="right", fontsize=11, weight="bold")
        ax.annotate(f"{avg[-1]:.0f}%  (≈{avg[-1]/avg[0]:.1f}× rise)", (avg_steps[-1], avg[-1]),
                    textcoords="offset points", xytext=(-6, 10), ha="right",
                    fontsize=11, weight="bold")
    ax.set_xlabel("RLVR training step (checkpoint)", fontsize=13)
    ax.set_ylabel("Verbalized eval-awareness rate (% of rollouts)", fontsize=13)
    ax.set_title("VEA across RLVR training, by benchmark", fontsize=16, pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    pct_top_axis(ax, 2300, label="% of final RLVR step (2300)")
    fig.tight_layout()
    fig.savefig(P.f("fig2_rlvr_vea_benchmarks.png"), dpi=140)
    print("saved ->", P.f("fig2_rlvr_vea_benchmarks.png"))


if __name__ == "__main__":
    main()
