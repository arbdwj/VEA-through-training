"""Where in the chain-of-thought does VEA appear?

Join VEA-positive judgments (which store the flagged QUOTE) with the rollout completions,
locate the quote inside the CoT (text before </think>), and report its position both as a
fraction of the CoT and in approximate tokens. RLVR sweep (richest VEA signal).
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")


def norm(s):
    """lowercase + collapse whitespace; strip wrapping quotes/space/punct."""
    s = (s or "").strip().strip('"\'` ').strip()
    return re.sub(r"\s+", " ", s.lower())


def cot_of(completion):
    c = completion or ""
    if "<think>" in c:
        c = c.split("<think>", 1)[1]
    if "</think>" in c:
        c = c.split("</think>", 1)[0]
    return c


def main():
    # 1) VEA-positive quotes keyed by (benchmark, step, prompt_id, rollout_id)
    quotes = {}
    for fp, force in [(os.path.join(OUT, "bench_vea_judgments.jsonl"), None),
                      (os.path.join(OUT, "vea_judgments.jsonl"), "fortress")]:
        for l in open(fp):
            j = json.loads(l)
            if j.get("ok") and j.get("vea") and j.get("quote"):
                bm = force or j.get("benchmark")
                quotes[(bm, j["step"], j["prompt_id"], j["rollout_id"])] = j["quote"]

    # 2) walk rollout files, locate each positive quote in its CoT
    fracs, tok_pos, cot_toks = [], [], []
    rows_csv = []
    n_pos = matched = short = unmatched = 0
    files = (glob.glob(os.path.join(OUT, "bench_rollouts_step_*.jsonl"))
             + glob.glob(os.path.join(OUT, "fortress_rollouts_step_*.jsonl")))
    for fp in files:
        is_fortress = "fortress_rollouts" in fp
        for l in open(fp):
            r = json.loads(l)
            bm = "fortress" if is_fortress else r.get("benchmark")
            key = (bm, r["step"], r["prompt_id"], r["rollout_id"])
            if key not in quotes:
                continue
            n_pos += 1
            cot = cot_of(r["completion"])
            cot_c = re.sub(r"\s+", " ", cot.lower())
            q_c = norm(quotes[key])
            ncot_tok = max(len(cot) // 4, 1)          # ~4 chars/token
            cot_toks.append(ncot_tok)
            if len(q_c) < 12:
                short += 1
                continue
            idx = cot_c.find(q_c)
            if idx < 0:                                # fallback: first 30 chars of quote
                idx = cot_c.find(q_c[:30]) if len(q_c) >= 30 else -1
            if idx < 0:
                unmatched += 1
                continue
            matched += 1
            frac = idx / max(len(cot_c), 1)
            fracs.append(100 * frac)
            tok_pos.append(frac * ncot_tok)
            rows_csv.append((bm, r["step"], round(100 * frac, 2),
                             round(frac * ncot_tok), ncot_tok))

    fracs = np.array(fracs); tok_pos = np.array(tok_pos); cot_toks = np.array(cot_toks)
    print(f"VEA-positive rollouts found in rollout files: {n_pos}")
    print(f"  quote located: {matched}  | quote too short to localize: {short}  | "
          f"unmatched: {unmatched}  (match rate {100*matched/max(n_pos,1):.0f}%)\n")
    if len(fracs):
        ps = [5, 10, 25, 50, 75, 90, 95]
        print("Position of VEA quote as % through the CoT (0 = start, 100 = just before </think>):")
        for p in ps:
            print(f"  p{p:<2} = {np.percentile(fracs, p):5.1f}%")
        print(f"  mean = {fracs.mean():.1f}%\n")
        print("Share of VEA quotes appearing in each part of the CoT:")
        for lo, hi in [(0, 10), (0, 25), (0, 50), (50, 100), (75, 100)]:
            share = 100 * np.mean((fracs >= lo) & (fracs < hi if hi < 100 else fracs <= hi))
            print(f"  {lo:>3}-{hi:<3}%: {share:5.1f}%")
        print(f"\nAbsolute position (approx tokens from CoT start):")
        for p in [10, 25, 50, 75, 90]:
            print(f"  p{p:<2} = {np.percentile(tok_pos, p):6.0f} tok")
        print(f"\nFor context, CoT length (approx tokens): "
              f"median {np.percentile(cot_toks,50):.0f}, p90 {np.percentile(cot_toks,90):.0f}")

    # 3) dump small derived CSV for the bundle figure
    csv_path = os.path.join(OUT, "vea_position.csv")
    with open(csv_path, "w") as f:
        f.write("benchmark,step,frac_pct,tok_pos,cot_tok\n")
        for r in rows_csv:
            f.write(",".join(map(str, r)) + "\n")
    print(f"\nwrote {len(rows_csv)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
