"""Figure 10 — eval-gaming at the FINAL checkpoint of each post-training stage.

For SFT-final (step 10000), DPO, and RLVR-final (step 2300), pooled over all 5 safety
benchmarks, two quantities per stage (gpt-5-mini judge for BOTH axes):
  • VEA rate                 = % of rollouts that verbalize eval-awareness
  • eval-gaming gap (pp)      = refuse | VEA  −  refuse | non-VEA
                               (how much MORE the model refuses once it has flagged a 'test')

Companion bar plot to the within-question trend (Fig 5): the trend shows the dynamics over
training; this shows where each stage's FINAL model lands. Reads data/{sft,dpo,rlvr}/refusal.jsonl.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P

STAGES = [("sft", 10000, "SFT-final\n(step 10k)"),
          ("dpo", 0, "DPO"),
          ("rlvr", 2300, "RLVR-final\n(step 2300)")]


def wald(k, n):
    p = k / n if n else 0.0
    return p, (p * (1 - p) / n) ** 0.5 if n else 0.0


def main():
    R = P.refusal_rows()
    R = [r for r in R if r.get("ok") and r.get("refusal") is not None and r.get("vea") is not None]

    vea_rate, vea_err, gap, gap_err, rv, rnv, ns = [], [], [], [], [], [], []
    for stage, step, _ in STAGES:
        sub = [r for r in R if r["stage"] == stage and int(r["step"]) == step]
        vea = [r for r in sub if r["vea"]]
        non = [r for r in sub if not r["vea"]]
        n = len(sub)
        p_v, se_v = wald(sum(bool(r["vea"]) for r in sub), n)            # VEA rate
        rv_p, rv_se = wald(sum(r["refusal"] for r in vea), len(vea))     # refuse | VEA
        rn_p, rn_se = wald(sum(r["refusal"] for r in non), len(non))     # refuse | non-VEA
        vea_rate.append(100 * p_v); vea_err.append(196 * se_v)
        gap.append(100 * (rv_p - rn_p)); gap_err.append(196 * (rv_se**2 + rn_se**2) ** 0.5)
        rv.append(100 * rv_p); rnv.append(100 * rn_p); ns.append((n, len(vea)))

    x = np.arange(len(STAGES)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 6))
    b1 = ax.bar(x - w / 2, vea_rate, w, yerr=vea_err, capsize=5, color="tab:red",
                edgecolor="black", lw=0.8, label="VEA rate")
    b2 = ax.bar(x + w / 2, gap, w, yerr=gap_err, capsize=5, color="tab:orange",
                edgecolor="black", lw=0.8, label="eval-gaming gap  (refuse|VEA − refuse|non-VEA)")
    for xi, v, e in zip(x - w / 2, vea_rate, vea_err):
        ax.text(xi, v + e + 0.6, f"{v:.1f}%", ha="center", fontsize=10, weight="bold")
    for xi, v, e in zip(x + w / 2, gap, gap_err):
        ax.text(xi, v + e + 0.6, f"{v:+.0f}pp", ha="center", fontsize=10, weight="bold")

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([s[2] for s in STAGES], fontsize=12)
    ax.set_ylabel("percent  /  percentage points", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(vea_rate), max(gap)) + 16)
    ax.legend(fontsize=11, loc="upper center", ncol=2, framealpha=0.95)
    ax.set_title("VEA and eval-gaming at each stage's final checkpoint", fontsize=16, pad=10)
    fig.tight_layout()
    out = P.f("fig10_eval_gaming_final_ckpts.png")
    fig.savefig(out, dpi=140)
    print("saved ->", out)
    for (stage, step, lab), vrate, g, a, b, (n, nv) in zip(
            STAGES, vea_rate, gap, rv, rnv, ns):
        print(f"  {stage:<4} VEA {vrate:4.1f}%  refuse|VEA {a:4.0f}%  refuse|nonVEA {b:4.0f}%  "
              f"gap {g:+4.0f}pp  (n={n}, vea={nv})")


if __name__ == "__main__":
    main()
