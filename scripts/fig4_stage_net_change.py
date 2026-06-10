"""Figure 4 — Net VEA change produced by each post-training stage (SFT, DPO, RLVR),
per benchmark + the overall mean of all 5 benchmarks.

Per-stage delta = (VEA at end of stage) - (VEA at start of stage):
  ΔSFT  = SFT(step10000) - SFT(step1000)
  ΔDPO  = DPO            - SFT(step10000)
  ΔRLVR = RLVR(step2300) - DPO
Reads ../data/stage_vea.jsonl, ../data/rlvr_bench_vea.jsonl, ../data/rlvr_fortress_vea.jsonl.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P

BENCH = ["fortress", "jbb", "advbench", "harmfulqa", "agentharm"]
LABELS = {"fortress": "Fortress", "jbb": "JailbreakBench", "advbench": "AdvBench",
          "harmfulqa": "HarmfulQA", "agentharm": "AgentHarm"}
COL = {"fortress": "tab:blue", "jbb": "tab:orange", "advbench": "tab:green",
       "harmfulqa": "tab:red", "agentharm": "tab:purple", "Overall": "black"}
STAGES = ["SFT", "DPO", "RLVR"]
SFT0, SFT1, RL1 = 1000, 10000, 2300


def load(fp, force_bm=None):
    rows = [json.loads(l) for l in open(fp)]
    if force_bm:
        for r in rows:
            r["benchmark"] = force_bm
    return [r for r in rows if r.get("ok")]


def rate(rows, bm, step=None):
    sel = [r for r in rows if r.get("benchmark") == bm and (step is None or r.get("step") == step)]
    return 100.0 * sum(bool(r["vea"]) for r in sel) / len(sel) if sel else np.nan


def main():
    stage = load(P.d("stage_vea.jsonl"))
    sft = [r for r in stage if r.get("stage") == "sft"]
    dpo = [r for r in stage if r.get("stage") == "dpo"]
    rlvr = load(P.d("rlvr_bench_vea.jsonl")) + load(P.d("rlvr_fortress_vea.jsonl"), force_bm="fortress")

    def anchors(bm):
        return {"sft0": rate(sft, bm, SFT0), "sft1": rate(sft, bm, SFT1),
                "dpo": rate(dpo, bm), "rl1": rate(rlvr, bm, RL1)}

    def deltas(a):
        return [a["sft1"] - a["sft0"], a["dpo"] - a["sft1"], a["rl1"] - a["dpo"]]

    series = {bm: deltas(anchors(bm)) for bm in BENCH}
    A = {k: np.nanmean([anchors(b)[k] for b in BENCH]) for k in ("sft0", "sft1", "dpo", "rl1")}
    series["Overall"] = deltas(A)

    order = BENCH + ["Overall"]
    x = np.arange(len(STAGES)) * 0.62
    n = len(order)
    w = 0.52 / n
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, key in enumerate(order):
        vals = series[key]
        off = (i - (n - 1) / 2) * w
        lbl = LABELS.get(key, key) + (" (mean)" if key == "Overall" else "")
        bars = ax.bar(x + off, vals, w, color=COL[key], label=lbl,
                      edgecolor="k" if key == "Overall" else "none",
                      linewidth=1.0 if key == "Overall" else 0, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.5 if v >= 0 else -0.5),
                    f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.5, rotation=90, color="dimgrey")

    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{STAGES[0]}\n(1k→10k)", f"{STAGES[1]}\n(vs SFT end)",
                        f"{STAGES[2]}\n(DPO→2300)"], fontsize=11)
    ax.set_ylabel("net Δ VEA  (percentage points)", fontsize=13)
    ax.set_title("Net VEA change by post-training stage", fontsize=16, pad=10)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=.25, axis="y")
    ax.legend(ncol=2, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(P.f("fig4_stage_net_change_v2.png"), dpi=140)
    print("saved ->", P.f("fig4_stage_net_change_v2.png"))


if __name__ == "__main__":
    main()
