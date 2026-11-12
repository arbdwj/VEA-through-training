"""Pretraining VEA: v1 judge (Gemini-2.5-flash) vs gpt-5-mini on the same rollouts."""
import collections
import glob
import json
import os

from vea import paths

KEY = lambda r: (r["step"], r["prompt_id"], r["rollout_id"])


def main():
    rows = [json.loads(l) for p in sorted(glob.glob(str(paths.DATA / "pretrain_rollouts" / "*.jsonl"))) for l in open(p)]
    total = collections.Counter(r["step"] for r in rows)
    cot = collections.Counter(r["step"] for r in rows if "</think>" in r["completion"])
    gem = {KEY(r): r for r in map(json.loads, open(paths.ARCHIVE_V1 / "pretrain_vea_judgments.jsonl")) if r.get("ok")}
    gpt = {KEY(r): r for r in map(json.loads, open(paths.OUTPUTS / "pretrain_vea_gpt5mini.jsonl")) if r.get("ok")}
    both = set(gem) & set(gpt)
    pc = lambda a, b: f"{100 * a / b:.2f}%" if b else "–"
    print(f"{'step':>8}{'has CoT':>9}{'Gemini |CoT':>13}{'gpt5m |CoT':>12}{'Gemini |all':>13}{'gpt5m |all':>12}{'agree':>8}")
    for s in sorted(total):
        ks = [k for k in both if k[0] == s]
        g = sum(bool(gem[k]["vea"]) for k in ks); o = sum(bool(gpt[k]["vea"]) for k in ks)
        ag = sum(bool(gem[k]["vea"]) == bool(gpt[k]["vea"]) for k in ks)
        print(f"{s:>8}{cot[s]:>9}{pc(g, len(ks)):>13}{pc(o, len(ks)):>12}{pc(g, total[s]):>13}{pc(o, total[s]):>12}{pc(ag, len(ks)):>8}")
    gv = {k for k in both if gem[k]["vea"]}; ov = {k for k in both if gpt[k]["vea"]}
    print(f"\npooled: n={len(both)}  Gemini VEA={len(gv)}  gpt-5-mini VEA={len(ov)}  both={len(gv & ov)}  "
          f"Gemini-only={len(gv - ov)}  gpt5mini-only={len(ov - gv)}")
