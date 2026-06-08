"""Figure 1 — VEA across PRETRAINING checkpoints of Olmo-3-1125-32B (stage1-step*).

VEA | CoT  = VEA hits / rollouts that produced a </think>  (among coherent CoTs)
VEA | all  = VEA hits / all 3552 rollouts                  (no-</think> = False)
CIs are by-prompt cluster bootstrap (rollouts nested in prompts; ICC ~0.45 -> naive CIs
would be ~2x too narrow). Reference (Haiku judge, 20480 ctx): released base `main` = 8.84%.

Reads ../data/pretrain_rollouts/*.jsonl + ../data/pretrain_vea_judgments.jsonl.
"""
import glob
import json
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P
from _stats import cluster_ci
from _axes import pct_top_axis

BASE_MAIN_VEA = 8.84


def main():
    # VEA-positive rollouts (step, prompt_id, rollout_id)
    vea_pos = set()
    for l in open(P.d("pretrain_vea_judgments.jsonl")):
        j = json.loads(l)
        if j.get("ok") and j.get("vea"):
            vea_pos.add((j["step"], j["prompt_id"], j["rollout_id"]))

    # per-step, per-prompt outcome lists (all rollouts, and CoT-only rollouts)
    g_all = defaultdict(lambda: defaultdict(list))
    g_cot = defaultdict(lambda: defaultdict(list))
    think = defaultdict(int)
    tot = defaultdict(int)
    for fp in glob.glob(P.d("pretrain_rollouts", "stage_rollouts_pretrain_step*.jsonl")):
        s = int(re.search(r"step(\d+)", fp).group(1))
        for l in open(fp):
            r = json.loads(l)
            tot[s] += 1
            o = 1 if (s, r["prompt_id"], r["rollout_id"]) in vea_pos else 0
            g_all[s][r["prompt_id"]].append(o)
            if "</think>" in r["completion"]:
                think[s] += 1
                g_cot[s][r["prompt_id"]].append(o)

    steps = sorted(tot)
    x = [s / 1000 for s in steps]
    cot = [cluster_ci(g_cot[s]) for s in steps]
    alc = [cluster_ci(g_all[s]) for s in steps]
    cot_rate = [c[0] for c in cot]
    cot_err = [[c[1] for c in cot], [c[2] for c in cot]]
    all_rate = [c[0] for c in alc]
    all_err = [[c[1] for c in alc], [c[2] for c in alc]]
    think_rate = [100 * think[s] / tot[s] for s in steps]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, cot_rate, yerr=cot_err, marker="o", ms=9, lw=2.4, capsize=4,
                color="tab:red", label="VEA | produced a CoT  (judged rollouts)", zorder=5)
    ax.errorbar(x, all_rate, yerr=all_err, marker="s", ms=7, lw=2.0, capsize=3,
                color="tab:blue", alpha=0.8, label="VEA | all rollouts  (no-CoT = False)", zorder=4)
    ax.axhline(BASE_MAIN_VEA, color="gray", ls="--", lw=1.6, alpha=0.8)
    ax.annotate(f"released base `main` ≈ {BASE_MAIN_VEA:.1f}%\n"
                "(end of pretraining; Haiku judge, 20480 ctx — reference only)",
                xy=(x[-1], BASE_MAIN_VEA), xytext=(x[0] + 20, BASE_MAIN_VEA + 0.6),
                fontsize=8.5, color="dimgray", va="bottom")
    ax.annotate("→ post-training (SFT) installs VEA, lifting it further",
                xy=(x[-1], BASE_MAIN_VEA), xytext=(x[0] + 20, BASE_MAIN_VEA - 1.3),
                fontsize=8.5, color="dimgray", style="italic")
    for xi, cr, tr in zip(x, cot_rate, think_rate):
        ax.annotate(f"{tr:.0f}%\n</think>", (xi, cr), textcoords="offset points",
                    xytext=(0, 16), ha="center", fontsize=7.5, color="darkred")

    ax.set_xlabel("pretraining step  (stage1, thousands)")
    ax.set_ylabel("Verbalized eval-awareness rate (%)")
    ax.set_ylim(-0.4, max(BASE_MAIN_VEA + 2.5, 4))
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", fontsize=9)
    ax.set_title("VEA is essentially absent during pretraining — it is installed by post-training\n"
                 "Olmo-3-1125-32B stage1 checkpoints · 444 prompts × 8 rollouts · 2048-tok cap · "
                 "Gemini-2.5-flash F1 · by-prompt bootstrap CIs", fontsize=10.5)
    pct_top_axis(ax, 656000, thousands=True, label="% of final pretraining step (656k)")
    fig.tight_layout()
    fig.savefig(P.f("fig1_pretrain_vea.png"), dpi=140)
    print("saved ->", P.f("fig1_pretrain_vea.png"))
    for s, c, a, tr in zip(steps, cot, alc, think_rate):
        print(f"  step {s:>7}  VEA|CoT {c[0]:4.1f}% [-{c[1]:.1f}/+{c[2]:.1f}]  "
              f"VEA|all {a[0]:4.2f}% [-{a[1]:.2f}/+{a[2]:.2f}]  </think> {tr:3.0f}%")


if __name__ == "__main__":
    main()
