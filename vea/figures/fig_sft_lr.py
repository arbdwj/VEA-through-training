"""VEA through SFT for both learning-rate runs (5e-5, 1e-4), ending with the released SFT (main).

All checkpoints sampled under the same pinned eval template, gpt-5-mini judge.
Reads data/sft/vea.jsonl (stage sft, lr 5e-5 / 1e-4; stage sft_merge = released main).
95% CIs: prompt-cluster bootstrap.
"""
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _paths as P

LR_C = {"5e-5": "#2a9d8f", "1e-4": "#7b4fb0"}
MAIN_C = "#1f6fb4"
B = 2000


def rate_ci(rows, rng):
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        d[r["prompt_id"]][0] += bool(r["vea"]); d[r["prompt_id"]][1] += 1
    M = np.array(list(d.values()), float)
    rate = lambda i: 100 * M[i, 0].sum() / M[i, 1].sum()
    bs = [rate(rng.integers(0, len(M), len(M))) for _ in range(B)]
    return rate(slice(None)), np.percentile(bs, [2.5, 97.5])


def main():
    rng = np.random.default_rng(0)
    sft = P.rows("vea.jsonl", "sft")
    by = collections.defaultdict(list)
    for r in sft:
        by[(r["stage"], r.get("lr"), r["step"])].append(r)
    main_v, main_ci = rate_ci(by[("sft_merge", "merge", 0)], rng)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    x_main = 11800
    ends = {}
    for k, (lr, c) in enumerate(LR_C.items()):
        steps = sorted(s for (st, l, s) in by if st == "sft" and l == lr)
        if not steps:
            continue
        vals = [rate_ci(by[("sft", lr, s)], rng) for s in steps]
        v = np.array([a for a, _ in vals]); lo = np.array([b[0] for _, b in vals]); hi = np.array([b[1] for _, b in vals])
        off = (k - 0.5) * 60                     # nudge so error bars don't overlap
        ax.fill_between(np.array(steps) + off, lo, hi, color=c, alpha=.12, lw=0)
        ax.plot(np.array(steps) + off, v, "-o", color=c, lw=2, ms=6, label=f"SFT lr {lr}", zorder=3)
        ends[lr] = (steps[-1] + off, v[-1])
        print(f"lr {lr}: " + "  ".join(f"{s}:{a:.1f}" for s, a in zip(steps, v)))
    for lr, (xe, ye) in ends.items():
        ax.plot([xe, x_main], [ye, main_v], ":", color=LR_C[lr], lw=1.3, zorder=2)
    ax.errorbar(x_main, main_v, yerr=[[main_v - main_ci[0]], [main_ci[1] - main_v]], fmt="D", color=MAIN_C, ms=9,
                capsize=3, zorder=4, label="released SFT (main)")
    ax.text(x_main + 250, main_v, f"{main_v:.1f}%", va="center", fontsize=11, weight="bold")
    print(f"released SFT main: {main_v:.1f} [{main_ci[0]:.1f}, {main_ci[1]:.1f}]")

    ax.axvspan(11000, 12900, color="0.94", zorder=0)
    has_last = any(st == "sft" and s == 10790 for (st, _, s) in by)
    ticks = list(range(1000, 11000, 1000)) + ([10790] if has_last else []) + [x_main]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" for t in range(1000, 11000, 1000)] + (["10790\n(last)"] if has_last else [])
                       + ["released\n(main)"], fontsize=10)
    ax.set_xlim(500, 12900)
    ax.set_ylim(0, 35)
    ax.set_xlabel("SFT step", fontsize=12)
    ax.set_ylabel("VEA rate (% of rollouts)", fontsize=12)
    ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=10.5, frameon=False)
    lrs = [lr for lr in LR_C if any(st == "sft" and l == lr for (st, l, _) in by)]
    ax.set_title(f"VEA through SFT (lr {', '.join(lrs)}) and the released SFT model", fontsize=14, pad=10)
    fig.text(0.5, 0.005, "444 prompts × 8 rollouts per checkpoint, same pinned eval template, gpt-5-mini judge. "
             "Bands/bars: 95% prompt-cluster bootstrap.", ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = P.f("fig_sft_lr.png")
    fig.savefig(out, dpi=140)
    print("saved ->", out)


if __name__ == "__main__":
    main()
