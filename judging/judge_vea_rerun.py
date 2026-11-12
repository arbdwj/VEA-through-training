"""JUDGING stage for the template-fixed rerun (generation: generation/gen_checkpoints.py).

Pulls olmo-outputs:/rerun_v2/<key>.jsonl to outputs/rerun_v2/, judges VEA with the
exact same call as judge_vea.py (Goodfire Appendix F1 rubric on the text before
</think>, openai/gpt-5-mini via OpenRouter, reasoning effort "low"), and writes
outputs/rerun_v2_judgments/<key>.jsonl (resumable; every row records the judge).

  python judging/judge_vea_rerun.py judge                 # all finished checkpoints
  python judging/judge_vea_rerun.py judge --keys sft_5e-5_step10000,rlvr_step_0050
  python judging/judge_vea_rerun.py compare               # old vs new VEA table

Requires OPENROUTER_API_KEY.
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import judge_vea as J  # noqa: E402  (reuse rubric, extract_cot, judge_one, client)

RAW = os.path.join(ROOT, "outputs", "rerun_v2")
OUT = os.path.join(ROOT, "outputs", "rerun_v2_judgments")
# Direct OpenAI API (the OpenRouter account is blocked from gpt-5-mini). Same model,
# same rubric, same effort as the original judge_vea.py runs.
MODEL = "gpt-5-mini"
EFFORT = J.REASONING_EFFORT      # "low"
MAX_CONSEC_FAIL = 25             # circuit breaker: stop if the account gets blocked mid-run


def make_client():
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    return OpenAI(max_retries=0)


def judge_one(client, cot):
    cot = (cot or "")[:J.MAX_COT_CHARS]
    prompt = J.JUDGE_PROMPT.replace("{cot}", cot)
    last = None
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=J.MAX_COMPLETION_TOKENS, reasoning_effort=EFFORT)
            content = r.choices[0].message.content or ""
            vea, quote = J.parse_quote(content)
            return {"ok": True, "vea": vea, "quote": quote, "error": None}
        except Exception as e:
            last = e
            msg = str(e)
            if "401" in msg or "deactivated" in msg or "blocked" in msg or "Policy" in msg:
                break                                   # account-level: don't retry
            time.sleep(min(2 ** attempt, 30))
    return {"ok": False, "vea": None, "quote": None, "error": f"{type(last).__name__}: {str(last)[:300]}"}
# judge the headline comparisons first
PRIORITY = ["sft_5e-5_step10000", "rlvr_step_0050", "rlvr_step_1050", "rlvr_step_1800",
            "rlvr_step_2300", "sft_5e-5_step01000", "sft_5e-5_step05000"]


def finished_keys():
    import modal
    d = modal.Dict.from_name("vea-gen-progress")
    return sorted(k for k, v in d.items() if k != "_run" and isinstance(v, dict) and v.get("status") == "done")


def pull(key):
    os.makedirs(RAW, exist_ok=True)
    dst = os.path.join(RAW, f"{key}.jsonl")
    if not os.path.exists(dst):
        subprocess.run(["modal", "volume", "get", "--force", "olmo-outputs",
                        f"rerun_v2/{key}.jsonl", RAW + "/"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def judge(keys, workers):
    os.makedirs(OUT, exist_ok=True)
    keys = sorted(keys, key=lambda k: (PRIORITY.index(k) if k in PRIORITY else 99, k))
    client = make_client()
    jobs, outs, locks, counts = [], {}, {}, collections.Counter()
    totals = collections.Counter()
    for k in keys:
        rows = [json.loads(l) for l in open(pull(k))]
        out_path = os.path.join(OUT, f"{k}.jsonl")
        done = set()
        if os.path.exists(out_path):
            for l in open(out_path):
                r = json.loads(l)
                if r.get("ok"):
                    done.add((r["prompt_id"], r["rollout_id"]))
        todo = [r for r in rows if (r["prompt_id"], r["rollout_id"]) not in done]
        totals[k] = len(rows)
        counts[k] = len(rows) - len(todo)
        outs[k] = open(out_path, "a")
        locks[k] = threading.Lock()
        jobs += [(k, r) for r in todo]
        print(f"{k}: {len(rows)} rollouts, {len(todo)} to judge")
    print(f"total to judge: {len(jobs)} with {workers} workers ({MODEL}, effort={EFFORT})")

    def work(k, r):
        return k, r, judge_one(client, J.extract_cot(r["completion"]))

    t0, last, consec_fail, n_fail = time.time(), 0.0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, k, r) for k, r in jobs]
        for f in as_completed(futs):
            k, r, res = f.result()
            if not res["ok"]:
                consec_fail += 1
                n_fail += 1
                if consec_fail >= MAX_CONSEC_FAIL:
                    print(f"ABORT: {consec_fail} consecutive failures; last error: {res['error']}", flush=True)
                    for ff in futs:
                        ff.cancel()
                    break
                continue                                  # failed rows are not written; rerun resumes them
            consec_fail = 0
            row = {kk: r.get(kk) for kk in ("stage", "lr", "step", "benchmark", "prompt_id", "rollout_id",
                                             "n_tokens", "finish_reason", "revision", "template_rev")}
            row.update(ok=True, vea=res["vea"], quote=res["quote"], judge=MODEL, reasoning_effort=EFFORT)
            with locks[k]:
                outs[k].write(json.dumps(row) + "\n")
                outs[k].flush()
            counts[k] += 1
            if time.time() - last > 30:
                last = time.time()
                el = time.time() - t0
                print(f"[{el/60:5.1f}m] " + "  ".join(f"{k.replace('sft_5e-5_','sft_')}:{counts[k]}/{totals[k]}"
                                                     for k in keys if counts[k] < totals[k]) or "finishing")
    for fh in outs.values():
        fh.close()
    print(f"done in {(time.time()-t0)/60:.1f} min ({n_fail} failed calls, will be retried on rerun)")


def _rate(rows):
    ok = [r for r in rows if r.get("ok")]
    return (100 * sum(bool(r["vea"]) for r in ok) / len(ok), len(ok)) if ok else (float("nan"), 0)


def compare():
    old = collections.defaultdict(list)
    for r in map(json.loads, open(os.path.join(ROOT, "archive/v1/data/stage_vea.jsonl"))):
        old[(r["stage"], r["step"])].append(r)
    for f in ("rlvr_bench_vea.jsonl", "rlvr_fortress_vea.jsonl"):
        for r in map(json.loads, open(os.path.join(ROOT, "archive", "v1", "data", f))):
            old[("rlvr", r["step"])].append(r)
    new = {}
    for p in sorted(glob.glob(os.path.join(OUT, "*.jsonl"))):
        rows = [json.loads(l) for l in open(p)]
        if rows:
            new[(rows[0]["stage"], rows[0]["step"])] = rows
    print(f"{'checkpoint':<16}{'old VEA':>10}{'new VEA':>10}{'change':>9}{'judged':>12}")
    print(f"{'dpo (ref)':<16}{_rate(old[('dpo', 0)])[0]:>9.1f}%{'—':>10}")
    for key in sorted(new, key=lambda x: (x[0] != "sft", x[1])):
        o, _ = _rate(old.get(key, []))
        n, nn = _rate(new[key])
        print(f"{key[0]+' '+str(key[1]):<16}{o:>9.1f}%{n:>9.1f}%{n-o:>+8.1f}{nn:>7}/3552")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["judge", "compare"])
    ap.add_argument("--keys", default="")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    if a.mode == "judge":
        ks = [k.strip() for k in a.keys.split(",") if k.strip()] or finished_keys()
        judge(ks, a.workers)
    compare()
