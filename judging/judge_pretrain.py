"""Re-judge the pretraining rollouts with gpt-5-mini (OpenAI API, effort "low") so Fig 1
uses the same judge as every other stage, and compare against the original Gemini-2.5-flash
labels (data/pretrain_vea_judgments.jsonl). Same rule as judge_pretrain_vea.py: only rollouts
containing </think> are judged; the rest are VEA=False by construction.

  python judging/judge_pretrain.py [--workers 64]
Writes outputs/pretrain_vea_gpt5mini.jsonl (resumable) and prints the comparison.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import judge_vea as J  # noqa: E402
from judge_vea_rerun import make_client, judge_one, MODEL, EFFORT  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "pretrain_vea_gpt5mini.jsonl")
KEY = lambda r: (r["step"], r["prompt_id"], r["rollout_id"])


def load_rollouts():
    rows = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "data", "pretrain_rollouts", "stage_rollouts_pretrain_step*.jsonl"))):
        step = int(re.search(r"step(\d+)", fp).group(1))
        for r in map(json.loads, open(fp)):
            r["step"] = step
            rows.append(r)
    return rows


def judge(workers):
    rows = load_rollouts()
    has_cot = [r for r in rows if "</think>" in r["completion"]]
    done = {KEY(r) for r in map(json.loads, open(OUT)) if r.get("ok")} if os.path.exists(OUT) else set()
    todo = [r for r in has_cot if KEY(r) not in done]
    print(f"{len(rows)} rollouts | {len(has_cot)} with </think> | {len(todo)} to judge ({MODEL}, effort={EFFORT})", flush=True)
    client, lock, fh = make_client(), threading.Lock(), open(OUT, "a")

    def work(r):
        res = judge_one(client, J.extract_cot(r["completion"]))
        if not res["ok"]:
            return 1
        rec = {"stage": "pretrain", "step": r["step"], "benchmark": r["benchmark"], "prompt_id": r["prompt_id"],
               "rollout_id": r["rollout_id"], "ok": True, "vea": res["vea"], "quote": res["quote"],
               "judge": MODEL, "reasoning_effort": EFFORT}
        with lock:
            fh.write(json.dumps(rec) + "\n"); fh.flush()
        return 0

    with ThreadPoolExecutor(workers) as ex:
        fails = sum(ex.map(work, todo))
    fh.close()
    print(f"failed: {fails} (rerun to resume)")
    return rows


def compare(rows):
    total = collections.Counter(r["step"] for r in rows)
    cot = collections.Counter(r["step"] for r in rows if "</think>" in r["completion"])
    gem = {KEY(r): r for r in map(json.loads, open(os.path.join(ROOT, "archive", "v1", "data", "pretrain_vea_judgments.jsonl"))) if r.get("ok")}
    gpt = {KEY(r): r for r in map(json.loads, open(OUT)) if r.get("ok")}
    both = set(gem) & set(gpt)
    print(f"\n{'step':>8}{'has CoT':>9}{'Gemini |CoT':>13}{'gpt5m |CoT':>12}{'Gemini |all':>13}{'gpt5m |all':>12}{'agree':>8}")
    for s in sorted(total):
        ks = [k for k in both if k[0] == s]
        g = sum(bool(gem[k]["vea"]) for k in ks); o = sum(bool(gpt[k]["vea"]) for k in ks)
        ag = sum(bool(gem[k]["vea"]) == bool(gpt[k]["vea"]) for k in ks)
        pc = lambda a, b: f"{100*a/b:.2f}%" if b else "–"
        print(f"{s:>8}{cot[s]:>9}{pc(g, len(ks)):>13}{pc(o, len(ks)):>12}{pc(g, total[s]):>13}{pc(o, total[s]):>12}{pc(ag, len(ks)):>8}")
    gv = {k for k in both if gem[k]["vea"]}; ov = {k for k in both if gpt[k]["vea"]}
    print(f"\npooled: n={len(both)}  Gemini VEA={len(gv)}  gpt-5-mini VEA={len(ov)}  both={len(gv & ov)}  "
          f"Gemini-only={len(gv - ov)}  gpt5mini-only={len(ov - gv)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    compare(judge(ap.parse_args().workers))
