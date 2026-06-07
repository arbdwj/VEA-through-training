"""VEA judge for the PRETRAINING-checkpoint rollouts (outputs/pretrain/*.jsonl).

Same Goodfire Appendix F1 rubric and extract_cot() as judge_vea.py. Judge model is
Gemini-2.0-flash (works for CoT/VEA judging and is ~10x cheaper than Haiku; gpt-5-mini
is blocked at the account level).

Only rollouts that actually produced a </think> are sent to the judge — a base-model
rollout with no </think> produced no chain of thought to verbalize eval-awareness in,
so it is VEA=False by construction (this is why step 0, with 0% </think>, is 0% VEA).
The plot script can therefore report BOTH:
  - VEA | has-CoT   (denominator = rollouts with </think>)
  - VEA | all       (denominator = all 3552, no-</think> counted False)

Writes outputs/pretrain_vea_judgments.jsonl (resumable). Requires OPENROUTER_API_KEY.

Usage:
  python src/judge_pretrain_vea.py            # judge all pretrain checkpoints
  JUDGE_MODEL=anthropic/claude-haiku-4.5 python src/judge_pretrain_vea.py   # fallback
"""
import glob
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_vea as J

# gemini-2.0-flash-001 was de-listed from OpenRouter (404); 2.5-flash is the current
# cheap Gemini that handles CoT/VEA judging without content-blocking (~$3 for all ckpts).
MODEL = os.environ.get("JUDGE_MODEL", "google/gemini-2.5-flash")
WORKERS = int(os.environ.get("JUDGE_WORKERS", "40"))
OUT = os.path.join(J.OUTPUTS, "pretrain_vea_judgments.jsonl")
PRETRAIN_DIR = os.path.join(J.OUTPUTS, "pretrain")


def main():
    client = J.make_client()
    files = sorted(glob.glob(os.path.join(PRETRAIN_DIR, "stage_rollouts_pretrain_step*.jsonl")))
    if not files:
        sys.exit(f"no pretrain rollout files in {PRETRAIN_DIR}")

    rows = []
    for fp in files:
        step = int(re.search(r"step(\d+)", fp).group(1))
        for l in open(fp):
            r = json.loads(l)
            r["step"] = step
            rows.append(r)
    has_cot = [r for r in rows if "</think>" in r["completion"]]
    print(f"{len(rows)} rollouts across {len(files)} checkpoints | {len(has_cot)} with </think> "
          f"(only these are judged) | judge={MODEL}")

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                j = json.loads(l)
            except json.JSONDecodeError:
                continue
            if j.get("ok"):
                done.add((j["step"], j["prompt_id"], j["rollout_id"]))
    todo = [r for r in has_cot if (r["step"], r["prompt_id"], r["rollout_id"]) not in done]
    print(f"done={len(done)} todo={len(todo)} workers={WORKERS}")
    if not todo:
        print("nothing to judge.")
        return

    lock = threading.Lock()
    n = {"d": 0, "v": 0, "fail": 0}
    t0 = time.time()

    def work(r):
        cot = J.extract_cot(r["completion"])
        res = J.judge_one(client, cot, "", MODEL)   # effort="" -> no reasoning param
        rec = {"stage": "pretrain", "step": r["step"], "benchmark": r["benchmark"],
               "prompt_id": r["prompt_id"], "rollout_id": r["rollout_id"],
               "ok": res["ok"], "vea": res["vea"], "quote": res["quote"], "judge": MODEL}
        with lock:
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n["d"] += 1
            if not res["ok"]:
                n["fail"] += 1
            elif res["vea"]:
                n["v"] += 1
            if n["d"] % 250 == 0 or n["d"] == len(todo):
                print(f"  {n['d']}/{len(todo)} | VEA {n['v']} | fails {n['fail']} | "
                      f"{n['d']/max(time.time()-t0,1e-6):.1f}/s")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in as_completed([ex.submit(work, r) for r in todo]):
            pass
    print(f"done in {time.time()-t0:.0f}s | VEA hits {n['v']} | fails {n['fail']}")


if __name__ == "__main__":
    main()
