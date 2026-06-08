"""Figure 3 — single continuous VEA plot across post-training: BASE -> SFT -> DPO -> RLVR.

Each stage's measured checkpoints are solid lines (per benchmark). BASE->SFT and SFT->DPO
transitions are DOTTED bridges (direction of change only — no intermediate checkpoints).
RLVR flows SOLID from DPO (it is initialized from DPO). BASE (step 0) judged by Haiku.

Reads ../data/stage_vea.jsonl, ../data/rlvr_bench_vea.jsonl, ../data/rlvr_fortress_vea.jsonl,
../data/base_vea_haiku.jsonl.
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P

BORDER = ["Fortress", "JailbreakBench", "AdvBench", "HarmfulQA", "AgentHarm"]
CANON = {"fortress": "Fortress", "jbb": "JailbreakBench", "advbench": "AdvBench",
         "harmfulqa": "HarmfulQA", "agentharm": "AgentHarm"}
COL = dict(zip(BORDER, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]))


def rate_table(rows):
    agg = {}
    for r in rows:
        if not r.get("ok"):
            continue
        bm = CANON.get(r.get("benchmark"), r.get("benchmark"))
        k = (bm, r["step"])
        h, n = agg.get(k, (0, 0))
        agg[k] = (h + int(bool(r["vea"])), n + 1)
    out = []
    for (bm, s), (h, n) in agg.items():
        p = h / n
        out.append({"benchmark": bm, "step": s, "vea": p * 100,
                    "ci": 1.96 * np.sqrt(p * (1 - p) / n) * 100})
    return pd.DataFrame(out)


def load(fp, force_bm=None):
    rows = [json.loads(l) for l in open(fp)]
    if force_bm:
        for r in rows:
            r["benchmark"] = force_bm
    return rows


def main():
    stage = load(P.d("stage_vea.jsonl"))
    sft = rate_table([r for r in stage if r.get("stage") == "sft"])
    dpo = rate_table([r for r in stage if r.get("stage") == "dpo"])
    rlvr = rate_table(load(P.d("rlvr_bench_vea.jsonl"))
                      + load(P.d("rlvr_fortress_vea.jsonl"), force_bm="fortress"))
    base = rate_table(load(P.d("base_vea_haiku.jsonl")))

    sft_steps = sorted(sft.step.unique())
    rlvr_steps = sorted(rlvr.step.unique())
    GAP1, GAP2 = 3.0, 1.5
    x_sft = {s: i for i, s in enumerate(sft_steps)}
    sft_end = len(sft_steps) - 0.5
    x_dpo = (len(sft_steps) - 1) + GAP1
    x_rlvr = {s: x_dpo + GAP2 + j for j, s in enumerate(rlvr_steps)}
    x_max = max(x_rlvr.values())
    dpo_center = (sft_end + x_dpo) / 2
    base_x = -2.0

    fig, ax = plt.subplots(figsize=(15.5, 7))
    ax.axvspan(base_x - 0.8, -0.5, color="#8c564b", alpha=0.06)
    ax.axvspan(-0.5, sft_end, color="tab:blue", alpha=0.05)
    ax.axvspan(sft_end, x_dpo, color="grey", alpha=0.10)
    ax.axvspan(x_dpo, x_max + 0.5, color="tab:green", alpha=0.05)

    for bm in BORDER:
        c = COL[bm]
        ds = sft[sft.benchmark == bm].sort_values("step")
        dr = rlvr[rlvr.benchmark == bm].sort_values("step")
        dd = dpo[dpo.benchmark == bm]
        if not len(ds) or not len(dr) or not len(dd):
            continue
        xs = [x_sft[s] for s in ds.step]
        xr = [x_rlvr[s] for s in dr.step]
        dpo_y = dd.vea.iloc[0]
        # per-point CIs omitted here (5 overlapping series -> clutter); effective n is small,
        # see Fig 1/2 for clustered CIs and the Limitations note in the text.
        ax.plot(xs, ds.vea, color=c, marker="o", ms=4, lw=1.9, label=bm)
        ax.plot([xs[-1], x_dpo], [ds.vea.iloc[-1], dpo_y], ls=":", color=c, lw=1.4, alpha=.7)
        ax.plot([x_dpo] + xr, [dpo_y] + list(dr.vea), ls="-", color=c, lw=1.9, zorder=2)
        ax.plot(xr, dr.vea, color=c, ls="none", marker="s", ms=5, zorder=3)
        ax.plot(x_dpo, dpo_y, marker="*", ms=20, color=c, markeredgecolor="k",
                markeredgewidth=.6, zorder=5)
        db_b = base[base.benchmark == bm]
        if len(db_b):
            base_y = db_b.vea.iloc[0]
            ax.plot([base_x, xs[0]], [base_y, ds.vea.iloc[0]], ls=":", color=c, lw=1.4, alpha=.7)
            ax.plot(base_x, base_y, marker="*", ms=20, color=c, markeredgecolor="k",
                    markeredgewidth=.6, zorder=5)

    ymax = 80
    ax.set_ylim(-6, ymax)
    ax.set_xlim(base_x - 1.0, x_max + 0.7)
    ax.set_yticks(range(0, ymax + 1, 10))
    ax.text(base_x, ymax * 0.97, "BASE\n(step 0)", ha="center", va="top", fontsize=10,
            weight="bold", color="#8c564b")
    ax.text(base_x, -3, "Haiku\njudge*", ha="center", va="center", fontsize=7,
            style="italic", color="#8c564b")
    ax.text((len(sft_steps) - 1) / 2, ymax * 0.97, "SFT  (Think-SFT, steps 1k→10k)",
            ha="center", va="top", fontsize=11, weight="bold", color="tab:blue")
    ax.text(dpo_center, ymax * 0.99, "DPO", ha="center", va="top", fontsize=12,
            weight="bold", color="dimgrey")
    ax.text(dpo_center, -3, "no intermediate\ncheckpoints", ha="center", va="center",
            fontsize=8, style="italic", color="dimgrey")
    ax.text((x_dpo + x_max) / 2, ymax * 0.97, "RLVR  (Think, steps 50→2300)",
            ha="center", va="top", fontsize=11, weight="bold", color="tab:green")

    ticks = [base_x, x_sft[sft_steps[0]], x_sft[sft_steps[len(sft_steps)//2]], x_sft[sft_steps[-1]],
             x_dpo,
             x_rlvr[rlvr_steps[0]], x_rlvr[rlvr_steps[len(rlvr_steps)//2]], x_rlvr[rlvr_steps[-1]]]
    def _pl(s, final):
        return f"{s}\n{100*s/final:.0f}%"
    labels = ["0", _pl(sft_steps[0], 10000), _pl(sft_steps[len(sft_steps)//2], 10000), _pl(sft_steps[-1], 10000),
              "DPO", _pl(rlvr_steps[0], 2300), _pl(rlvr_steps[len(rlvr_steps)//2], 2300), _pl(rlvr_steps[-1], 2300)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("post-training step  (% = of that stage's final step; stages sequential, not to scale)")
    ax.set_ylabel("VEA rate (% of rollouts)")
    ax.grid(alpha=.25, axis="y")
    ax.legend(title="benchmark", loc="upper right", fontsize=9)
    ax.set_title("Verbalized eval-awareness across OLMo-3 32B Think post-training:  "
                 "BASE → SFT → DPO → RLVR\nall 5 benchmarks · same 444 prompts · Goodfire F1 judge",
                 fontsize=12)
    fig.text(0.5, 0.005,
             "★ = single checkpoints (BASE step 0, DPO).  Dotted lines = unmeasured stage transitions "
             "(BASE→SFT and SFT→DPO), shown as direction-of-change only.  RLVR flows SOLID from DPO "
             "(initialized from it).   Per-point CIs omitted (5 overlapping series); effective n≈200 "
             "per point (ICC≈0.45), see text.   *BASE judged by claude-haiku-4.5 (gpt-5-mini "
             "unavailable) and tends to under-detect — a conservative floor, not strictly comparable.",
             ha="center", fontsize=8, style="italic", color="dimgrey")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(P.f("fig3_stage_connected.png"), dpi=140)
    print("saved ->", P.f("fig3_stage_connected.png"))


if __name__ == "__main__":
    main()
