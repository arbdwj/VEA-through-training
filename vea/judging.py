"""LLM-judge plumbing: client, single calls, and a resumable parallel runner.

Rows that fail (API error) are not written, so re-running resumes exactly where it stopped.
"""
import collections
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from vea import rubrics as R

FATAL = ("401", "deactivated", "blocked", "Policy")   # account-level errors: don't retry


def make_client():
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    return OpenAI(max_retries=0)


def _call(client, jcfg, prompt, max_tokens):
    last = None
    for attempt in range(6):
        try:
            r = client.chat.completions.create(model=jcfg["model"], messages=[{"role": "user", "content": prompt}],
                                               max_completion_tokens=max_tokens,
                                               reasoning_effort=jcfg["reasoning_effort"])
            return True, r.choices[0].message.content or ""
        except Exception as e:
            last = e
            if any(s in str(e) for s in FATAL):
                break
            time.sleep(min(2 ** attempt, 30))
    return False, f"{type(last).__name__}: {str(last)[:300]}"


def vea_one(client, jcfg, completion):
    cot = R.extract_cot(completion)[:R.MAX_COT_CHARS]
    ok, out = _call(client, jcfg, R.VEA_PROMPT.replace("{cot}", cot), 4000)
    if not ok:
        return {"ok": False, "error": out}
    vea, quote = R.parse_quote(out)
    return {"ok": True, "vea": vea, "quote": quote}


def refusal_one(client, jcfg, completion, request):
    ans = R.answer_of(completion)
    if not ans:
        return {"ok": True, "refusal": None, "no_answer": True}
    prompt = R.REFUSAL_PROMPT.replace("{q}", request[:R.MAX_REQUEST_CHARS]).replace("{t}", ans[:R.MAX_ANSWER_CHARS])
    ok, out = _call(client, jcfg, prompt, 2000)
    return {"ok": True, "refusal": R.parse_refusal(out), "no_answer": False} if ok else {"ok": False, "error": out}


def row_key(r):
    return (r["step"], r["prompt_id"], r["rollout_id"])


def run(tasks, judge_fn, make_row, jcfg, label="judging"):
    """tasks: {out_path: [raw rows]}. judge_fn(client, raw) -> result; make_row(raw, result) -> output row."""
    client = make_client()
    todo, fhs, locks, counts, totals = [], {}, {}, collections.Counter(), collections.Counter()
    for out_path, rows in tasks.items():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        done = set()
        if os.path.exists(out_path):
            done = {row_key(r) for r in map(json.loads, open(out_path)) if r.get("ok")}
        pending = [r for r in rows if row_key(r) not in done]
        name = os.path.basename(out_path)[:-6]
        totals[name], counts[name] = len(rows), len(rows) - len(pending)
        fhs[out_path], locks[out_path] = open(out_path, "a"), threading.Lock()
        todo += [(out_path, r) for r in pending]
        print(f"{name}: {len(rows)} rows, {len(pending)} to judge")
    print(f"{label}: {len(todo)} calls, {jcfg['workers']} workers ({jcfg['model']}, effort={jcfg['reasoning_effort']})",
          flush=True)

    t0, last, consec, n_fail = time.time(), 0.0, 0, 0
    with ThreadPoolExecutor(max_workers=jcfg["workers"]) as ex:
        futs = {ex.submit(judge_fn, client, r): (p, r) for p, r in todo}
        for f in as_completed(futs):
            p, r = futs[f]
            res = f.result()
            if not res["ok"]:
                consec += 1; n_fail += 1
                if consec >= jcfg["max_consecutive_failures"]:
                    print(f"ABORT: {consec} consecutive failures; last error: {res['error']}", flush=True)
                    for ff in futs:
                        ff.cancel()
                    break
                continue
            consec = 0
            with locks[p]:
                fhs[p].write(json.dumps(make_row(r, res)) + "\n"); fhs[p].flush()
            counts[os.path.basename(p)[:-6]] += 1
            if time.time() - last > 30:
                last = time.time()
                print(f"[{(time.time() - t0) / 60:5.1f}m] " + ("  ".join(
                    f"{k}:{counts[k]}/{totals[k]}" for k in totals if counts[k] < totals[k]) or "finishing"), flush=True)
    for fh in fhs.values():
        fh.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min ({n_fail} failed calls; re-run to retry them)")


# ---- raw rollouts on the Modal outputs volume ----

def pull(key, cfg, root):
    """Fetch <modal.out_dir>/<key>.jsonl from the outputs volume into paths.raw (if not already local)."""
    raw_dir = os.path.join(root, cfg["paths"]["raw"])
    dst = os.path.join(raw_dir, f"{key}.jsonl")
    if not os.path.exists(dst):
        os.makedirs(raw_dir, exist_ok=True)
        vol_path = cfg["modal"]["out_dir"].removeprefix("/outputs/").strip("/")
        subprocess.run(["modal", "volume", "get", "--force", cfg["modal"]["outputs_volume"],
                        f"{vol_path}/{key}.jsonl", raw_dir + "/"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def finished_keys(cfg):
    import modal
    d = modal.Dict.from_name(cfg["modal"]["progress_dict"])
    return sorted(k for k, v in d.items() if k != "_run" and isinstance(v, dict) and v.get("status") == "done")
