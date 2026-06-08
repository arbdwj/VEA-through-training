"""Figure 5 — WITHIN-question eval-gaming trend on Fortress, per checkpoint.

At each checkpoint, restrict to Fortress questions that elicit VEA (>=2 VEA & >=2 non-VEA
rollouts out of 8), average each question's refuse|VEA and refuse|non-VEA, plot over
training. This controls for the prompt (removes the between-question confound), so it is
the honest per-question eval-gaming signal. SFT (top) / RLVR (bottom).
Reads ../data/refusal_haiku.jsonl.
"""
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P
from _axes import pct_top_axis

MIN = 2


def main():
    R = [json.loads(l) for l in open(P.d("refusal_haiku.jsonl"))]
    R = [r for r in R if r.get("ok") and not r.get("no_answer")
         and r.get("refusal") is not None and r.get("vea") is not None
         and r.get("benchmark") == "fortress"]
    for r in R:
        r["refusal"] = bool(r["refusal"]); r["vea"] = bool(r["vea"])

    def series(stage):
        steps = sorted({r["step"] for r in R if r["stage"] == stage})
        out = {"steps": [], "v": [], "ve": [], "nv": [], "nve": [], "n": []}
        for s in steps:
            q = defaultdict(lambda: {"v": [], "nv": []})
            for r in R:
                if r["stage"] == stage and r["step"] == s:
                    q[r["prompt_id"]]["v" if r["vea"] else "nv"].append(r["refusal"])
            sel = [(np.mean(g["v"]), np.mean(g["nv"])) for g in q.values()
                   if len(g["v"]) >= MIN and len(g["nv"]) >= MIN]
            if not sel:
                continue
            vv = np.array([x[0] for x in sel]) * 100
            nn = np.array([x[1] for x in sel]) * 100
            out["steps"].append(s)
            out["v"].append(vv.mean()); out["ve"].append(1.96 * vv.std(ddof=1) / np.sqrt(len(vv)))
            out["nv"].append(nn.mean()); out["nve"].append(1.96 * nn.std(ddof=1) / np.sqrt(len(nn)))
            out["n"].append(len(sel))
        return out

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    for ax, stage, title in [(axes[0], "sft", "SFT  (Think-SFT)"),
                             (axes[1], "rlvr", "RLVR  (Think)")]:
        d = series(stage)
        st = d["steps"]
        ax.fill_between(st, d["nv"], d["v"], color="orange", alpha=0.13, label="eval-gaming gap")
        ax.errorbar(st, d["v"], yerr=d["ve"], marker="o", ms=7, lw=2.2, capsize=3,
                    color="tab:red", label="refuse | VEA (within-question)")
        ax.errorbar(st, d["nv"], yerr=d["nve"], marker="s", ms=7, lw=2.2, capsize=3,
                    color="tab:blue", label="refuse | non-VEA (within-question)")
        for x, gv, gn, n in zip(st, d["v"], d["nv"], d["n"]):
            ax.annotate(f"+{gv-gn:.0f}", (x, (gv+gn)/2), ha="center", va="center",
                        fontsize=9, color="darkorange", weight="bold")
            ax.annotate(f"n={n}", (x, gv), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7, color="tab:red")
        ax.set_title(title, fontsize=12); ax.set_ylabel("refusal rate (%)")
        ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.set_xlabel("training step")
        pct_top_axis(ax, 10000 if stage == "sft" else 2300,
                     label=f"% of final {stage.upper()} step ({'10k' if stage=='sft' else '2300'})")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle("WITHIN-question eval-gaming trend on Fortress  (questions that elicit VEA)\n"
                 "per-question refuse|VEA vs refuse|non-VEA, ≥2/≥2 rollouts · OLMo-3 32B Think · "
                 "Haiku F2 · small n → wide CIs", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(P.f("fig5_within_question_eval_gaming.png"), dpi=140)
    print("saved ->", P.f("fig5_within_question_eval_gaming.png"))


if __name__ == "__main__":
    main()
