"""MODEL vs DATA: VEA at the SFT and DPO stages, for OLMo's own outputs and for the
training responses of each stage (same gpt-5-mini judge on both sides).

MODEL: SFT step 10000 vs released DPO, all 5 benchmarks pooled (stage_vea.jsonl).
DATA:  6,735 safety prompts shared by the SFT and DPO sets: SFT target (DeepSeek-R1)
       vs DPO chosen (qwen3-32b); values from the matched-triples figure (n=6735 each).
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P

DATA_SFT, DATA_DPO, N_DATA = 21.0, 8.0, 6735


def rate(rows):
    p = np.mean([bool(r["vea"]) for r in rows])
    return 100 * p, 196 * np.sqrt(p * (1 - p) / len(rows))


def main():
    st = [r for r in P.rows("vea.jsonl", "sft", "dpo") if r.get("ok")]
    m_sft = rate([r for r in st if r["stage"] == "sft" and r["step"] == 10000])
    m_dpo = rate([r for r in st if r["stage"] == "dpo"])
    d_sft = (DATA_SFT, 196 * np.sqrt(.21 * .79 / N_DATA))
    d_dpo = (DATA_DPO, 196 * np.sqrt(.08 * .92 / N_DATA))

    fig, ax = plt.subplots(figsize=(8.4, 6))
    x, w = np.array([0, 1.0]), 0.36
    for i, (lab, col, vals) in enumerate([("SFT stage", "tab:blue", [m_sft, d_sft]),
                                          ("DPO stage", "tab:orange", [m_dpo, d_dpo])]):
        xs = x + (i - .5) * w
        ax.bar(xs, [v[0] for v in vals], w, yerr=[v[1] for v in vals], capsize=6, color=col,
               edgecolor="k", label=lab, zorder=3)
        for xx, (v, e) in zip(xs, vals):
            ax.text(xx, v + e + .6, f"{v:.0f}%", ha="center", fontsize=12, weight="bold")
    for j, (a, b) in enumerate([(m_sft, m_dpo), (d_sft, d_dpo)]):
        ax.annotate("", xy=(x[j] + w / 2, b[0] + b[1] + 2.5), xytext=(x[j] - w / 2 + .03, a[0] + a[1] + 2.5),
                    arrowprops=dict(arrowstyle="->", color="dimgrey", lw=1.6, connectionstyle="arc3,rad=-0.3"))
        word = "rose to" if b[0] > a[0] else "shrank to"
        ax.text(x[j], max(a[0], b[0]) + 9, f"{word}\n{100*b[0]/a[0]:.0f}% of SFT", ha="center",
                fontsize=11, color="dimgrey", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["MODEL\n(OLMo's own outputs)", "DATA\n(the training responses)"], fontsize=12)
    ax.set_ylabel("Verbalized eval-awareness rate (%)", fontsize=11)
    ax.set_ylim(0, 40)
    ax.grid(axis="y", alpha=.3, zorder=0)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_title("DPO data has less VEA, but the DPO model does not", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(P.f("fig_model_vs_data_vea.png"), dpi=140)
    print(f"model SFT10k {m_sft[0]:.1f}% -> DPO {m_dpo[0]:.1f}% | data {DATA_SFT}% -> {DATA_DPO}%")


if __name__ == "__main__":
    main()
