"""New figure: the training-vs-eval template gap in VEA shrinks during RLVR.

Top: VEA under the eval template (prompt ends <think>) and under the training template
(SFT: prompt ends assistant\\n; model writes <think>\\n itself) / one-token swap (RLVR:
eval template with final '>' -> '>\\n'), per checkpoint.
Bottom: ratio (training or swap) / eval with 95% paired prompt-cluster bootstrap CIs.
SFT 5e-5 steps: ours (original run = training template; rerun = eval template).
SFT merge, DPO: numbers from the revisiting post (hollow markers, no CI available).
RLVR: ours, one-token swap at every step with a judged swap file; training template at 2300.
"""
import collections, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _paths as P

ROOT = P.ROOT
J = os.path.join(ROOT, "outputs/rerun_v2_judgments")
EVAL_C, TRAIN_C, SWAP_C = "#1f6fb4", "#d9711a", "#8e44ad"
RLVR_STEPS = [50, 300, 550, 800, 1050, 1300, 1550, 1800, 2050, 2300]


def rd(p):
    return [json.loads(l) for l in open(p)]


def per_prompt(rows):
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("ok"):
            d[r["prompt_id"]][0] += bool(r["vea"]); d[r["prompt_id"]][1] += 1
    return d


def paired(tr, ev, B=2000, seed=0):
    a, b = per_prompt(tr), per_prompt(ev)
    Pq = sorted(set(a) & set(b), key=str)
    A = np.array([a[p] for p in Pq], float); E = np.array([b[p] for p in Pq], float)
    f = lambda i: (100 * A[i, 0].sum() / A[i, 1].sum(), 100 * E[i, 0].sum() / E[i, 1].sum())
    t, e = f(np.arange(len(Pq)))
    rng = np.random.default_rng(seed)
    bs = np.array([f(rng.integers(0, len(Pq), len(Pq))) for _ in range(B)])
    ci = lambda x: np.percentile(x, [2.5, 97.5])
    return dict(train=t, eval=e, train_ci=ci(bs[:, 0]), eval_ci=ci(bs[:, 1]),
                ratio=t / e, ratio_ci=ci(bs[:, 0] / bs[:, 1]), n_prompts=len(Pq))


def rlvr_eval(step):
    """Eval-template rows for an RLVR step: rerun file if one exists, else data_v2."""
    p = os.path.join(J, f"rlvr_step_{step:04d}.jsonl")
    if os.path.exists(p):
        return rd(p)
    D2 = os.path.join(ROOT, "data_v2")
    rows = [r for r in rd(os.path.join(D2, "rlvr_bench_vea.jsonl")) if r["step"] == step]
    rows += [dict(r, prompt_id=f"fortress_{r['prompt_id']}")
             for r in rd(os.path.join(D2, "rlvr_fortress_vea.jsonl")) if r["step"] == step]
    return rows


def eb(ax, x, d, key, **kw):
    v, c = d[key], d[key + "_ci"]
    ax.errorbar(x, v, yerr=[[v - c[0]], [c[1] - v]], capsize=2, zorder=3, **kw)


def main():
    stage = rd(os.path.join(ROOT, "data/stage_vea.jsonl"))  # original run = training template for SFT
    sft = []
    for s in range(1000, 10001, 1000):
        tr = [r for r in stage if r["stage"] == "sft" and r["step"] == s]
        sft.append((f"{s//1000}k", paired(tr, rd(os.path.join(J, f"sft_5e-5_step{s:05d}.jsonl")))))
    ext = [("SFT\nmerge*", dict(train=45.0, eval=20.6)), ("DPO*", dict(train=44.8, eval=22.5))]
    rl = []
    for s in RLVR_STEPS:
        p = os.path.join(J, f"rlvrswap_step_{s:04d}.jsonl")
        if os.path.exists(p):
            rl.append((s, paired(rd(p), rlvr_eval(s))))
    tp = os.path.join(J, "rlvrtrain_step_2300.jsonl")
    r_train = paired(rd(tp), rlvr_eval(2300)) if os.path.exists(tp) else None

    # x positions: SFT 0..9, gap, merge/DPO, gap, RLVR steps
    xs_sft = np.arange(10.0)
    xs_ext = np.array([10.6, 11.6])
    xs_rl = 12.8 + np.arange(len(rl))
    labels = [l for l, _ in sft] + [l for l, _ in ext] + [str(s) for s, _ in rl]
    ticks = np.concatenate([xs_sft, xs_ext, xs_rl])

    w = 10 + 0.45 * max(0, len(rl) - 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(w, 8.4), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.15, 1]))

    # --- top: VEA under each template
    for i, (_, d) in enumerate(sft):
        x = xs_sft[i]
        ax1.plot([x, x], [d["eval"], d["train"]], color="0.78", lw=1.5, zorder=1)
        eb(ax1, x, d, "eval", fmt="o", color=EVAL_C, ms=7,
           label="eval template (ends <think>)" if i == 0 else None)
        eb(ax1, x, d, "train", fmt="s", color=TRAIN_C, ms=7,
           label="SFT training template (ends assistant\\n)" if i == 0 else None)
    for x, (_, d) in zip(xs_ext, ext):
        ax1.plot([x, x], [d["eval"], d["train"]], color="0.78", lw=1.5, ls=":", zorder=1)
        ax1.plot(x, d["eval"], "o", mfc="white", mec=EVAL_C, mew=2, ms=7, zorder=3)
        ax1.plot(x, d["train"], "s", mfc="white", mec=TRAIN_C, mew=2, ms=7, zorder=3)
    for j, (x, (s, d)) in enumerate(zip(xs_rl, rl)):
        ax1.plot([x, x], [d["eval"], d["train"]], color="0.78", lw=1.5, zorder=1)
        eb(ax1, x, d, "eval", fmt="o", color=EVAL_C, ms=7)
        eb(ax1, x, d, "train", fmt="D", color=SWAP_C, ms=6.5,
           label="one-token swap (<think> → <think>\\n)" if j == 0 else None)
    if r_train is not None and rl and rl[-1][0] == 2300:
        eb(ax1, xs_rl[-1] + 0.25, r_train, "train", fmt="s", color=TRAIN_C, ms=6, alpha=.8)
    ax1.set_ylabel("VEA rate (% of rollouts)", fontsize=12)
    ax1.set_ylim(0, 60); ax1.grid(axis="y", alpha=.3)
    ax1.legend(loc="lower left", fontsize=9.5, frameon=True)
    ax1.set_title("The template gap in VEA shrinks during RLVR", fontsize=15, pad=10)

    # --- bottom: ratio
    sft_r = [d["ratio"] for _, d in sft]
    ax2.axhspan(min(sft_r), max(sft_r), color=TRAIN_C, alpha=.10, zorder=0)
    ax2.text(xs_sft[0] - 0.4, max(sft_r) + 0.04, f"SFT range {min(sft_r):.1f}–{max(sft_r):.1f}×", fontsize=9, color="0.35")
    ax2.axhline(1, color="0.4", lw=1, ls="--")
    ax2.text(xs_sft[0] - 0.4, 1.03, "no template effect", fontsize=9, color="0.4")
    for x, (_, d) in zip(xs_sft, sft):
        eb(ax2, x, d, "ratio", fmt="s", color=TRAIN_C, ms=7)
    for x, (_, d) in zip(xs_ext, ext):
        ax2.plot(x, d["train"] / d["eval"], "s", mfc="white", mec=TRAIN_C, mew=2, ms=7)
    if rl:
        ax2.plot(xs_rl, [d["ratio"] for _, d in rl], color=SWAP_C, lw=1.2, alpha=.6, zorder=2)
        for x, (_, d) in zip(xs_rl, rl):
            eb(ax2, x, d, "ratio", fmt="D", color=SWAP_C, ms=6.5)
        if r_train is not None and rl[-1][0] == 2300:
            eb(ax2, xs_rl[-1] + 0.25, r_train, "ratio", fmt="s", color=TRAIN_C, ms=6, alpha=.8)
    ax2.set_ylabel("VEA ratio  (training or swap ÷ eval)", fontsize=12)
    ax2.set_ylim(0.8, 2.6); ax2.grid(axis="y", alpha=.3)
    ax2.set_xticks(ticks); ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.text(xs_sft.mean(), -0.16, "SFT (lr 5e-5) step", transform=ax2.get_xaxis_transform(), ha="center", fontsize=10)
    if rl:
        ax2.text(xs_rl.mean(), -0.16, "RLVR step", transform=ax2.get_xaxis_transform(), ha="center", fontsize=10)
    ax2.text(xs_ext.mean(), -0.25, "* revisiting post's numbers", transform=ax2.get_xaxis_transform(),
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout()
    out = P.f("fig_template_gap_rlvr.png")
    fig.savefig(out, dpi=140)
    print("saved ->", out)
    print(f"SFT ratio range {min(sft_r):.2f}-{max(sft_r):.2f}")
    for s, d in rl:
        print(f"RLVR {s:>4}: eval {d['eval']:.1f}  swap {d['train']:.1f}  ratio {d['ratio']:.2f} "
              f"[{d['ratio_ci'][0]:.2f},{d['ratio_ci'][1]:.2f}]  prompts {d['n_prompts']}")
    if r_train is not None:
        print(f"RLVR 2300 training template ratio {r_train['ratio']:.2f} {np.round(r_train['ratio_ci'], 2)}")


if __name__ == "__main__":
    main()
