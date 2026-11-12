"""VEA before vs after DPO: (A) released SFT merge -> DPO, (B) SFT 5e-5 step 10k -> DPO.

Both checkpoints sampled under the same pinned eval template, judged by gpt-5-mini.
Left: VEA rate per benchmark and overall (95% prompt-cluster bootstrap CIs).
Right: DPO - SFT merge difference with a paired prompt-level bootstrap CI.
Reads data/sft/vea.jsonl (stage sft_merge) and data/dpo/vea.jsonl.
"""
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _paths as P

MERGE_C, DPO_C, S5_C = "#1f6fb4", "#d9711a", "#2a9d8f"
BENCH = ["advbench", "agentharm", "fortress", "harmfulqa", "jbb"]
B = 4000


def per_prompt(rows):
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("ok"):
            d[r["prompt_id"]][0] += bool(r["vea"]); d[r["prompt_id"]][1] += 1
    return d


def paired(a_rows, b_rows, rng):
    a, b = per_prompt(a_rows), per_prompt(b_rows)
    Pq = sorted(set(a) & set(b))
    A = np.array([a[p] for p in Pq], float); Bm = np.array([b[p] for p in Pq], float)
    rate = lambda M, i: 100 * M[i, 0].sum() / M[i, 1].sum()
    ra, rb = rate(A, slice(None)), rate(Bm, slice(None))
    bs = np.array([(rate(A, i), rate(Bm, i)) for i in (rng.integers(0, len(Pq), len(Pq)) for _ in range(B))])
    diff = bs[:, 1] - bs[:, 0]
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    ci = lambda x: np.percentile(x, [2.5, 97.5])
    return dict(a=ra, b=rb, a_ci=ci(bs[:, 0]), b_ci=ci(bs[:, 1]), d=rb - ra, d_ci=ci(diff), p=p, n=len(Pq))


def compare(before, after, rng):
    groups = [("all 444 prompts", None)] + [(b, b) for b in BENCH]
    out = []
    for label, bm in groups:
        sel = (lambda rows: rows) if bm is None else (lambda rows, bm=bm: [r for r in rows if r["benchmark"] == bm])
        out.append((label, paired(sel(before), sel(after), rng)))
    return out


def draw_row(axo, axb, res, before_label, before_short, before_c, title):
    w = 0.38
    o = res[0][1]

    def bars(ax, x, r, label_vals):
        for dx, v, ci, c, lab in ((-w / 2 - 0.01, r["a"], r["a_ci"], before_c, before_label),
                                  (w / 2 + 0.01, r["b"], r["b_ci"], DPO_C, "after DPO")):
            ax.bar(x + dx, v, w, color=c, edgecolor="white", linewidth=1, label=lab, zorder=2)
            ax.errorbar(x + dx, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="0.25", capsize=3, lw=1.2, zorder=3)
            if label_vals:
                ax.text(x + dx, ci[1] + 1.2, f"{v:.1f}%", ha="center", fontsize=11, weight="bold")

    def ptxt(p):
        return "p<0.001" if p < 0.001 else (f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}")

    bars(axo, 0.0, o, True)
    axo.set_xticks([-w / 2 - 0.01, w / 2 + 0.01]); axo.set_xticklabels([before_short, "DPO"], fontsize=11)
    axo.set_xlim(-0.7, 0.7)
    axo.text(0, max(o["a_ci"][1], o["b_ci"][1]) + 6,
             f"Δ {o['d']:+.1f} pts\n95% CI [{o['d_ci'][0]:+.1f}, {o['d_ci'][1]:+.1f}]\n{ptxt(o['p'])}",
             ha="center", fontsize=10, color="0.2", weight="bold" if o["p"] < 0.05 else None)
    axo.set_ylabel("VEA rate (% of rollouts)", fontsize=12)
    axo.set_title("Overall (444 prompts)", fontsize=11.5, color="0.3")
    xb = np.arange(len(BENCH), dtype=float)
    for i, (_, r) in enumerate(res[1:]):
        bars(axb, xb[i], r, False)
        axb.text(xb[i], max(r["a_ci"][1], r["b_ci"][1]) + 1.5, f"{r['d']:+.1f} pts\n{ptxt(r['p'])}", ha="center",
                 fontsize=9.5, color="0.2", weight="bold" if r["p"] < 0.05 else None)
    axb.set_xticks(xb); axb.set_xticklabels(BENCH, fontsize=11)
    axb.set_title("By benchmark", fontsize=11.5, color="0.3")
    h, l = axb.get_legend_handles_labels()
    axb.legend(h[:2], l[:2], loc="lower right", fontsize=10.5, frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.12))
    for ax in (axo, axb):
        ax.grid(axis="y", alpha=.3, zorder=0); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axo.set_ylim(0, 58)
    axo.annotate(title, xy=(0, 1.15), xycoords="axes fraction", fontsize=13.5, weight="bold", ha="left")


def main():
    rng = np.random.default_rng(0)
    sft = P.rows("vea.jsonl", "sft")
    merge = [r for r in sft if r["stage"] == "sft_merge"]
    s10k = [r for r in sft if r["stage"] == "sft" and r["lr"] == "5e-5" and r["step"] == 10000]
    dpo = P.rows("vea.jsonl", "dpo")
    rows = [(compare(merge, dpo, rng), "SFT merge (released SFT, DPO's parent)", "SFT merge", MERGE_C,
             "A. Released SFT merge → DPO"),
            (compare(s10k, dpo, rng), "SFT lr 5e-5, step 10k", "SFT 5e-5\nstep 10k", S5_C,
             "B. SFT lr 5e-5 step 10k → DPO")]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), gridspec_kw=dict(width_ratios=[1, 2.6], hspace=0.55),
                             sharey=True)
    for (res, bl, bs, bc, title), (axo, axb) in zip(rows, axes):
        draw_row(axo, axb, res, bl, bs, bc, title)
    fig.suptitle("VEA before vs after DPO", fontsize=15, y=0.995)
    fig.text(0.5, 0.005, "Same 444 prompts × 8 rollouts, same pinned eval template, gpt-5-mini judge. "
             "Error bars: 95% prompt-cluster bootstrap; Δ and p: paired bootstrap over prompts.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = P.f("fig_merge_vs_dpo.png")
    fig.savefig(out, dpi=140)
    print("saved ->", out)
    for (res, bl, *_ ) in rows:
        print(bl)
        for label, r in res:
            print(f"  {label:<16} before {r['a']:5.1f}%  DPO {r['b']:5.1f}%  diff {r['d']:+5.1f} "
                  f"[{r['d_ci'][0]:+.1f},{r['d_ci'][1]:+.1f}]  p={r['p']:.3f}  n_prompts={r['n']}")

if __name__ == "__main__":
    main()
