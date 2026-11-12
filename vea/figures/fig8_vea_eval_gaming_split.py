"""Figure 8 — Of the rollouts that verbalize eval-awareness, how many then refuse
('eval-gaming-consistent') vs comply, across RLVR — and is that refusal above the
non-VEA baseline (i.e. genuine eval-gaming) or just baseline safety refusal?

(A) VEA-positive rollouts per RLVR step, split into refused vs complied (counts).
(B) refusal rate among VEA vs non-VEA rollouts per step; the shaded gap is the pooled
    eval-gaming increment. Pooled over all 5 benchmarks → an UPPER BOUND (VEA-eliciting
    prompts are also the refusal-prone ones); the within-question version (Fig 5) is the
    confound-controlled estimate. Reads ../data/refusal_haiku.jsonl.
"""
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P
from _axes import step_pct_labels

from collections import defaultdict as _dd
import os as _os
R = [json.loads(l) for l in open(P.d(_os.environ.get("REFUSAL_FILE", "refusal_gpt5mini.jsonl")))]
R = [r for r in R if r.get("stage") == "rlvr" and r.get("ok") and not r.get("no_answer")
     and r.get("vea") is not None and r.get("refusal") is not None]
# only steps with full benchmark coverage (the extra refusal run added Fortress-only steps,
# which belong to Fig 5's within-question trend, not this all-benchmark pooled view)
_bench = _dd(set)
for r in R:
    _bench[r["step"]].add(r["benchmark"])
steps = sorted(s for s in _bench if len(_bench[s]) >= 5)
R = [r for r in R if r["step"] in steps]
if not steps:
    import sys as _sys
    print("fig8 skipped: no RLVR step has refusal labels on all 5 benchmarks in this data dir "
          "(corrected data/ has Fortress-only refusal labels for re-sampled steps; "
          "run with DATA_DIR=archive/v1/data for the as-published version)")
    _sys.exit(0)

vea_ref, vea_comp, nonvea_rate, vea_rate = [], [], [], []
for s in steps:
    sub = [r for r in R if r["step"] == s]
    vea = [r for r in sub if r["vea"]]
    nonvea = [r for r in sub if not r["vea"]]
    vr = sum(1 for r in vea if r["refusal"])
    vea_ref.append(vr)
    vea_comp.append(len(vea) - vr)
    vea_rate.append(100 * vr / max(len(vea), 1))
    nonvea_rate.append(100 * sum(1 for r in nonvea if r["refusal"]) / max(len(nonvea), 1))

x = np.arange(len(steps))
fig, (a, b) = plt.subplots(2, 1, figsize=(7.5, 9))

# (A) stacked counts
a.bar(x, vea_ref, color="tab:red", label="refused  (eval-gaming-consistent)")
a.bar(x, vea_comp, bottom=vea_ref, color="tab:gray", alpha=0.6,
      label="complied  (flagged 'test' but answered)")
for i in range(len(steps)):
    tot = vea_ref[i] + vea_comp[i]
    a.text(x[i], tot + 25, f"{vea_rate[i]:.0f}% refused", ha="center", fontsize=9, weight="bold")
    a.text(x[i], vea_ref[i] / 2, str(vea_ref[i]), ha="center", va="center", color="white", fontsize=9)
    a.text(x[i], vea_ref[i] + vea_comp[i] / 2, str(vea_comp[i]), ha="center", va="center", fontsize=8)
a.set_xticks(x); a.set_xticklabels(step_pct_labels(steps, 2300), fontsize=8)
a.set_xlabel("RLVR training step  (% of final, 2300)"); a.set_ylabel("VEA-positive rollouts")
a.set_ylim(0, max(np.array(vea_ref) + np.array(vea_comp)) * 1.15)
a.set_title("(A) Of rollouts that verbalize eval-awareness, how many then refuse?", fontsize=13)
a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.25, axis="y")

# (B) VEA vs non-VEA refusal rate (the eval-gaming contrast)
b.fill_between(x, nonvea_rate, vea_rate, color="orange", alpha=0.18, label="eval-gaming gap (pooled)")
b.plot(x, vea_rate, "o-", color="tab:red", lw=2.2, ms=8, label="refuse | VEA")
b.plot(x, nonvea_rate, "s-", color="tab:blue", lw=2.2, ms=8, label="refuse | non-VEA")
for i in range(len(steps)):
    b.annotate(f"+{vea_rate[i]-nonvea_rate[i]:.0f}", (x[i], (vea_rate[i]+nonvea_rate[i])/2),
               ha="center", va="center", fontsize=9, color="darkorange", weight="bold")
b.set_xticks(x); b.set_xticklabels(step_pct_labels(steps, 2300), fontsize=9)
b.set_xlabel("RLVR training step  (% of final, 2300)", fontsize=12)
b.set_ylabel("refusal rate (%)", fontsize=12)
b.set_ylim(0, 100)
b.set_title("(B) Is it eval-gaming or baseline safety? VEA vs non-VEA refusal (pooled = upper bound)",
            fontsize=13)
b.legend(fontsize=10, loc="center right"); b.grid(alpha=0.25)

fig.suptitle("VEA and eval-gaming across RLVR", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(P.f("fig8_vea_eval_gaming_split.png"), dpi=140)
print("saved ->", P.f("fig8_vea_eval_gaming_split.png"))
for i, s in enumerate(steps):
    print(f"  step {s:>5}: VEA refused={vea_ref[i]} complied={vea_comp[i]} "
          f"({vea_rate[i]:.0f}%) | non-VEA refusal={nonvea_rate[i]:.0f}% | gap=+{vea_rate[i]-nonvea_rate[i]:.0f}")
