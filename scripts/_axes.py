"""Helpers to show '% of final step' alongside the absolute training step on plots.

Percentage convention matches the fraction-of-training definition: pct = 100 * step / final_step
(RLVR final = 2300, SFT final = 10000, pretraining final = 656000).
"""

FINAL = {"rlvr": 2300, "sft": 10000, "pretrain": 656000}


def pct_top_axis(ax, final, thousands=False, label="% of final step"):
    """Add a secondary top x-axis that re-labels the existing (step-valued) x-axis as a
    percentage of `final`. Use when the x data are real step values (thousands=True if the
    x-axis is in thousands of steps)."""
    div = (final / 1000.0) if thousands else float(final)
    sec = ax.secondary_xaxis("top", functions=(lambda x: 100.0 * x / div,
                                               lambda p: p * div / 100.0))
    sec.set_xlabel(label, fontsize=9)
    return sec


def step_pct_labels(steps, final):
    """Tick labels 'step\\n(NN%)' for categorically-positioned step axes."""
    return [f"{s}\n({100.0 * s / final:.0f}%)" for s in steps]
