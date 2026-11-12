"""Is the BROAD VEA-position distribution a between-prompt confound?

Decompose the spread of VEA position into within-prompt vs between-prompt components
(same logic as the rate-ICC analysis). If between-prompt dominates, each prompt has a
characteristic VEA location and the broad pooled histogram is a mixture artifact.

Also: does absolute VEA token-position track PROMPT length (the "model restates the prompt
first" confound)?  Writes outputs/vea_position_full.csv (adds prompt_id, prompt_tok).
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")


def norm(s):
    s = (s or "").strip().strip('"\'` ').strip()
    return re.sub(r"\s+", " ", s.lower())


def cot_of(c):
    c = c or ""
    if "<think>" in c:
        c = c.split("<think>", 1)[1]
    if "</think>" in c:
        c = c.split("</think>", 1)[0]
    return c


def main():
    quotes = {}
    for fp, force in [(os.path.join(OUT, "bench_vea_judgments.jsonl"), None),
                      (os.path.join(OUT, "vea_judgments.jsonl"), "fortress")]:
        for l in open(fp):
            j = json.loads(l)
            if j.get("ok") and j.get("vea") and j.get("quote"):
                bm = force or j.get("benchmark")
                quotes[(bm, j["step"], j["prompt_id"], j["rollout_id"])] = j["quote"]

    rows = []   # (bm, step, prompt_id, frac, tok_pos, cot_tok, prompt_tok)
    files = (glob.glob(os.path.join(OUT, "bench_rollouts_step_*.jsonl"))
             + glob.glob(os.path.join(OUT, "fortress_rollouts_step_*.jsonl")))
    for fp in files:
        isf = "fortress_rollouts" in fp
        for l in open(fp):
            r = json.loads(l)
            bm = "fortress" if isf else r.get("benchmark")
            key = (bm, r["step"], r["prompt_id"], r["rollout_id"])
            if key not in quotes:
                continue
            cot = cot_of(r["completion"])
            cot_c = re.sub(r"\s+", " ", cot.lower())
            q_c = norm(quotes[key])
            if len(q_c) < 12:
                continue
            idx = cot_c.find(q_c)
            if idx < 0:
                idx = cot_c.find(q_c[:30]) if len(q_c) >= 30 else -1
            if idx < 0:
                continue
            frac = idx / max(len(cot_c), 1)
            # calibrate to REAL tokens using vLLM's n_tokens for this completion:
            # CoT's token share = n_tokens * (CoT chars / completion chars)
            comp = r["completion"] or ""
            ntok = r.get("n_tokens") or max(len(comp) // 4, 1)
            cot_tok_real = ntok * len(cot) / max(len(comp), 1)
            tok_first = frac * cot_tok_real                       # first-VEA-token index
            ptok = max(len(r.get("prompt", "")) // 4, 1)
            rows.append((bm, r["step"], r["prompt_id"], 100 * frac,
                         round(tok_first), round(cot_tok_real), ptok))

    with open(os.path.join(OUT, "vea_position_full.csv"), "w") as f:
        f.write("benchmark,step,prompt_id,frac_pct,tok_first,cot_tok,prompt_tok\n")
        for r in rows:
            f.write(",".join(str(x) for x in (r[0], r[1], r[2], round(r[3], 2),
                                              r[4], r[5], r[6])) + "\n")

    frac = np.array([r[3] for r in rows])
    tok = np.array([r[4] for r in rows])
    clen = np.array([r[5] for r in rows])
    plen = np.array([r[6] for r in rows])
    print(f"matched VEA rollouts: {len(rows)}")
    print(f"\nFIRST VEA token index (calibrated to real n_tokens):")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{p:<2} = {np.percentile(tok, p):5.0f} tok")
    print(f"  mean = {tok.mean():.0f} tok")
    for thr in [100, 200, 300, 500]:
        print(f"  share with first VEA token <= {thr}: {100*np.mean(tok<=thr):.0f}%")

    # ---- within vs between prompt variance of frac_pct ----
    byp = defaultdict(list)
    for r in rows:
        byp[(r[0], r[2])].append(r[3])     # group by (benchmark, prompt_id)
    multi = {k: v for k, v in byp.items() if len(v) >= 3}
    grand = frac.mean()
    pm = np.array([np.mean(v) for v in multi.values()])
    # one-way random-effects style decomposition
    within_var = np.mean([np.var(v, ddof=1) for v in multi.values()])
    between_var = np.var(pm, ddof=1)
    icc = between_var / (between_var + within_var)
    print(f"\nWithin- vs between-prompt spread of VEA position (% of CoT), "
          f"{len(multi)} prompts w/ >=3 VEA rollouts:")
    print(f"  overall SD (pooled)      : {frac.std():.1f} pts")
    print(f"  within-prompt SD (typical): {np.sqrt(within_var):.1f} pts")
    print(f"  between-prompt SD        : {np.sqrt(between_var):.1f} pts")
    print(f"  position ICC (between/total): {icc:.2f}   "
          f"-> {'between-prompt dominates (prompt is a confound)' if icc>0.4 else 'mostly within-prompt'}")
    print(f"  spread of per-prompt MEAN positions: p10={np.percentile(pm,10):.0f}%, "
          f"median={np.percentile(pm,50):.0f}%, p90={np.percentile(pm,90):.0f}%")

    # ---- prompt-length confound on absolute position ----
    print(f"\nPrompt-length confound:")
    print(f"  r(VEA token-pos, prompt length) = {np.corrcoef(tok, plen)[0,1]:+.2f}")
    print(f"  r(VEA token-pos, CoT length)    = {np.corrcoef(tok, clen)[0,1]:+.2f}")
    print(f"  prompt length (tok): median {np.median(plen):.0f}, p90 {np.percentile(plen,90):.0f}")


if __name__ == "__main__":
    main()
