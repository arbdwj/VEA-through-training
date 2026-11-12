"""Figure 9 — VEA in the training-data traces the model learns from:
SFT target (DeepSeek-R1) vs DPO chosen (qwen3-32b), on the safety subsets.
The gap is a FREQUENCY effect — conditional on appearing, VEA intensity is equal (median 5/10).
Reads ../data/sft_vea_safety.jsonl + ../data/dpo_vea_safety.jsonl.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P


def rate(path, side=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("ok") and (side is None or r.get("side") == side)]
    n = len(rows); v = sum(bool(r["vea"]) for r in rows)
    p = v / n if n else 0.0
    return 100 * p, 100 * 1.96 * np.sqrt(p * (1 - p) / n) if n else 0.0, n


def main():
    sft, sft_ci, n_sft = rate(P.d("sft_vea_safety.jsonl"))
    dch, dch_ci, n_dch = rate(P.d("dpo_vea_safety.jsonl"), side="chosen")
    drj, drj_ci, n_drj = rate(P.d("dpo_vea_safety.jsonl"), side="rejected")

    labels = [f"SFT target\n(DeepSeek-R1)\nn={n_sft}",
              f"DPO chosen\n(qwen3-32b)\nn={n_dch}",
              f"DPO rejected\n(qwen3-0.6b)\nn={n_drj}"]
    vals, cis = [sft, dch, drj], [sft_ci, dch_ci, drj_ci]

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    bars = ax.bar(labels, vals, yerr=cis, capsize=6,
                  color=["tab:purple", "tab:green", "tab:gray"], alpha=0.85, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=13, weight="bold")
    ax.set_ylabel("traces verbalizing eval-awareness (%)", fontsize=13)
    ax.set_ylim(0, max(vals) + 8)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("VEA in the training data", fontsize=16, pad=10)
    fig.tight_layout()
    fig.savefig(P.f("fig9_sft_dpo_data_vea.png"), dpi=140)
    print("saved ->", P.f("fig9_sft_dpo_data_vea.png"))
    print(f"  SFT {sft:.1f}% | DPO chosen {dch:.1f}% | DPO rejected {drj:.1f}%")


if __name__ == "__main__":
    main()
