"""Figure 6 — When does the FIRST verbalized-eval-awareness token appear in the CoT?

For each VEA-positive RLVR rollout, a cheap Gemini pass returns the EARLIEST eval-aware span,
which is then located and counted with the EXACT OLMo-3 tokenizer (offset mapping). This is
the accurate 'first VEA token' (not the original judge's representative quote, not chars/4).
Reads ../data/vea_first_token.csv (built by src/refine_vea_first_token.py).

(A) distribution of the first-VEA-token index (exact tokens), with cumulative markers.
(B) vs CoT length: first-VEA-token vs CoT length (hexbin) + median per length-decile.
(C) vs RLVR checkpoint: median first-VEA-token (tokens) and position (% of CoT) per step.
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths as P
from _axes import step_pct_labels

step, ft, clen, frac = [], [], [], []
with open(P.d("vea_first_token.csv")) as f:
    for row in csv.DictReader(f):
        step.append(int(row["step"])); ft.append(float(row["first_tok"]))
        clen.append(float(row["cot_tok"])); frac.append(float(row["frac_pct"]))
step = np.array(step); ft = np.array(ft); clen = np.array(clen); frac = np.array(frac)
med = np.median(ft)
r_tok = np.corrcoef(clen, ft)[0, 1]
r_frac = np.corrcoef(clen, frac)[0, 1]

fig, (a, b, c) = plt.subplots(3, 1, figsize=(7.5, 14.5))

# (A) absolute first-VEA-token distribution
a.hist(ft[ft < 800], bins=np.arange(0, 801, 25), color="tab:purple", alpha=0.85,
       edgecolor="white")
a.axvline(med, color="k", ls="--", lw=1.6)
a.text(med + 8, a.get_ylim()[1] * 0.92, f"median {med:.0f} tok", fontsize=10)
for thr, lbl in [(100, "27%"), (200, "53%"), (300, "73%")]:
    a.axvline(thr, color="gray", ls=":", lw=1)
    a.text(thr + 4, a.get_ylim()[1] * (0.55), f"≤{thr}: {lbl}", fontsize=8, color="dimgray", rotation=90)
a.set_xlabel("first VEA token index in CoT  (exact OLMo-3 tokenizer)")
a.set_ylabel("VEA-positive rollouts")
a.set_title("(A) First VEA token: median ~187, half within the first 200 tokens")
a.grid(alpha=0.25, axis="y")

# (B) vs CoT length
m = clen < 1200
b.hexbin(clen[m], ft[m], gridsize=40, cmap="Blues", mincnt=1)
b.plot([0, 1200], [0, 1200], "k--", lw=1, alpha=.6, label="y = x  (end of CoT)")
edges = np.percentile(clen, np.arange(0, 101, 10))
xs, ys = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (clen >= lo) & (clen < hi)
    if sel.sum() > 20:
        xs.append((lo + hi) / 2); ys.append(np.median(ft[sel]))
b.plot(xs, ys, "o-", color="tab:red", lw=2, ms=5, label="median first VEA token")
b.set_xlim(0, 1200); b.set_ylim(0, 1200)
b.set_xlabel("CoT length (exact tokens)")
b.set_ylabel("first VEA token index")
b.set_title(f"(B) vs CoT length: appears early, only weakly later in longer CoTs\n"
            f"r(first-tok,len)={r_tok:.2f},  r(%pos,len)={r_frac:+.2f}")
b.legend(fontsize=8, loc="upper left")

# (C) vs RLVR checkpoint
steps = sorted(set(step.tolist()))
mt = [np.median(ft[step == s]) for s in steps]
mf = [np.median(frac[step == s]) for s in steps]
c.plot(steps, mt, "s-", color="tab:green", lw=2)
c.set_xticks(steps); c.set_xticklabels(step_pct_labels(steps, 2300), fontsize=7)
c.set_xlabel("RLVR training step  (% of final, 2300)"); c.set_ylabel("median first VEA token", color="tab:green")
c.tick_params(axis="y", labelcolor="tab:green"); c.set_ylim(0, 320)
c2 = c.twinx()
c2.plot(steps, mf, "o--", color="tab:purple", lw=2)
c2.set_ylabel("median position (% of CoT)", color="tab:purple")
c2.tick_params(axis="y", labelcolor="tab:purple"); c2.set_ylim(0, 60)
c.set_title("(C) vs RLVR checkpoint: first-VEA-token position is ~stable")
c.grid(alpha=0.25)

fig.suptitle("When the first verbalized-eval-awareness token appears in the chain-of-thought\n"
             f"OLMo-3.1-32B-Think RLVR · {len(ft):,} VEA rollouts · Gemini earliest-span + exact "
             "OLMo-3 tokenizer", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(P.f("fig6_vea_position_in_cot.png"), dpi=140)
print("saved ->", P.f("fig6_vea_position_in_cot.png"))
print(f"median first VEA token = {med:.0f}  | r(tok,len)={r_tok:.2f} r(%,len)={r_frac:.2f}")
