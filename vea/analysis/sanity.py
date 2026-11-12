"""Sanity check of the current pipeline: one checkpoint per stage (configs/sanity/*.yaml),
re-generated and re-judged end to end, compared with the existing data for the same checkpoint.

VEA rate = VEA-positive rollouts / all rollouts (pretrain: only rollouts that close </think>
are judged, the rest count as non-VEA, as in Fig 1). Difference CI: paired prompt-level bootstrap.
"consistent" = the 95% CI of (new - existing) contains 0.
"""
import collections
import json
import os

import numpy as np

from vea import config as C, paths

STAGES = ("pretrain", "sft", "dpo", "rlvr")
REF_RAW = {   # existing raw rollouts for the same checkpoint (generation stats, exact-match rate)
    "pretrain": paths.DATA / "pretrain" / "rollouts" / "stage_rollouts_pretrain_step0656000.jsonl",
    "sft": paths.OUTPUTS / "rerun_v2" / "sft_5e-5_step10000.jsonl",
    "dpo": paths.OUTPUTS / "old_raw" / "stage_rollouts_dpo_main.jsonl",
    "rlvr": paths.OUTPUTS / "rerun_v2" / "rlvr_step_2300.jsonl",
}
rd = lambda p: [json.loads(l) for l in open(p)]
ID = lambda r: (r["prompt_id"], r["rollout_id"])


def per_prompt(raw, vea_pos):
    d = collections.defaultdict(lambda: [0, 0])
    for r in raw:
        d[r["prompt_id"]][0] += ID(r) in vea_pos
        d[r["prompt_id"]][1] += 1
    return d


def gen_stats(raw):
    toks = sorted(r["n_tokens"] for r in raw)
    return {"close_think": 100 * np.mean(["</think>" in r["completion"] for r in raw]),
            "capped": 100 * np.mean([r["finish_reason"] == "length" for r in raw]),
            "median_tok": toks[len(toks) // 2]}


def main(B=2000):
    st = C.settings()
    rng = np.random.default_rng(0)
    print(f"{'stage':<9}{'checkpoint':<22}{'existing':>9}{'new':>8}{'diff [95% CI]':>22}  {'verdict':<11}"
          f"{'</think> ex/new':>16}{'capped ex/new':>15}{'median tok':>14}{'same text':>11}")
    for stage in STAGES:
        m = C.load_model(f"sanity/{stage}")
        step, key = m["steps"][0], C.keys(m)[0]
        raw_p = paths.ROOT / st["paths"]["raw"] / f"{key}.jsonl"
        lab_p = paths.ROOT / st["paths"]["vea"] / f"{key}.jsonl"
        if not (raw_p.exists() and lab_p.exists()):
            print(f"{stage:<9}{key:<22}  not generated/judged yet"); continue
        new_raw, ref_raw = rd(raw_p), rd(REF_RAW[stage])
        new_pos = {ID(r) for r in rd(lab_p) if r.get("ok") and r["vea"]}
        ref_pos = {ID(r) for r in rd(paths.DATA / stage / "vea.jsonl")
                   if r["stage"] == stage and r["step"] == step and r.get("ok") and r["vea"]}
        a, b = per_prompt(ref_raw, ref_pos), per_prompt(new_raw, new_pos)
        P = sorted(set(a) & set(b), key=str)
        A = np.array([a[p] for p in P], float); N = np.array([b[p] for p in P], float)
        rate = lambda X, i: 100 * X[i, 0].sum() / X[i, 1].sum()
        ex, nw = rate(A, slice(None)), rate(N, slice(None))
        bs = []
        for _ in range(B):
            i = rng.integers(0, len(P), len(P)); bs.append(rate(N, i) - rate(A, i))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        gs_e, gs_n = gen_stats(ref_raw), gen_stats(new_raw)
        ref_txt = {ID(r): r["completion"] for r in ref_raw}
        same = 100 * np.mean([ref_txt.get(ID(r)) == r["completion"] for r in new_raw])
        print(f"{stage:<9}{key:<22}{ex:>8.1f}%{nw:>7.1f}%{nw - ex:>+9.1f} [{lo:+.1f},{hi:+.1f}]  "
              f"{'consistent' if lo <= 0 <= hi else 'DIFFERENT':<11}"
              f"{gs_e['close_think']:>7.0f}/{gs_n['close_think']:<4.0f}%{gs_e['capped']:>9.1f}/{gs_n['capped']:<4.1f}%"
              f"{gs_e['median_tok']:>7}/{gs_n['median_tok']:<6}{same:>8.1f}%")
        bms = sorted({r["benchmark"] for r in new_raw})
        line = "   per benchmark (existing -> new): "
        for bm in bms:
            e = 100 * np.mean([ID(r) in ref_pos for r in ref_raw if r["benchmark"] == bm])
            n = 100 * np.mean([ID(r) in new_pos for r in new_raw if r["benchmark"] == bm])
            line += f"{bm} {e:.1f}->{n:.1f}  "
        print(line)
