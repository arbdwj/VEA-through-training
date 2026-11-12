"""Does the training-vs-eval template gap in VEA shrink during RLVR?  (pre-specified)

Gap at a checkpoint = VEA(training template: no system, no trailing <think>)
                      - VEA(eval template: generic system + trailing <think>).
PRIMARY: ratio train/eval at RLVR 2300 vs the SFT-era ratio (SFT 5e-5 steps 1000-10000, pooled).
"Shrinks" = RLVR-2300 ratio's 95% CI lies entirely below the SFT-era ratio's 95% CI.
Also: absolute gap, and how often the model opens <think> itself under the training template.
Paired prompt-cluster bootstrap (same 444 prompts in both conditions).
"""
import collections, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J, RAW = os.path.join(ROOT, "outputs/rerun_v2_judgments"), os.path.join(ROOT, "outputs")


def per_prompt(rows):
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("ok"):
            d[r["prompt_id"]][0] += bool(r["vea"]); d[r["prompt_id"]][1] += 1
    return d


def rd(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def eval_rows(step):
    for key in (f"rlvr_step_{step:04d}",):
        r = rd(os.path.join(J, f"{key}.jsonl"))
        if r:
            return r
    rows = [r for r in rd(os.path.join(ROOT, "data_v2/rlvr_bench_vea.jsonl")) if r["step"] == step]
    rows += [dict(r, benchmark="fortress") for r in rd(os.path.join(ROOT, "data_v2/rlvr_fortress_vea.jsonl")) if r["step"] == step]
    return rows


def self_open(key):
    r = rd(os.path.join(RAW, "old_raw", f"stage_rollouts_{key}.jsonl")) or rd(os.path.join(RAW, "rerun_v2", f"{key}.jsonl"))
    return None if not r else 100 * np.mean([x["completion"].lstrip().startswith("<think>") for x in r])


PAIRS = {}  # name -> (train rows, eval rows, raw key for self-open)
for s in range(1000, 10001, 1000):
    tr = [r for r in rd(os.path.join(ROOT, "data/stage_vea.jsonl")) if r["stage"] == "sft" and r["step"] == s]
    PAIRS[f"SFT {s}"] = (tr, rd(os.path.join(J, f"sft_5e-5_step{s:05d}.jsonl")), f"sft_5e-5_step{s:05d}")
for s in (50, 800, 1550, 2300):
    tr = rd(os.path.join(J, f"rlvrtrain_step_{s:04d}.jsonl"))
    if tr:
        PAIRS[f"RLVR {s}"] = (tr, eval_rows(s), f"rlvrtrain_step_{s:04d}")

rng = np.random.default_rng(0)
print(f"{'checkpoint':<11}{'train':>8}{'eval':>8}{'gap (pts)':>18}{'ratio':>18}{'self-open <think> (train)':>28}")
res = {}
for name, (tr, ev, key) in PAIRS.items():
    a, b = per_prompt(tr), per_prompt(ev)
    P = sorted(set(a) & set(b), key=str)
    A = np.array([a[p] for p in P], float); Bv = np.array([b[p] for p in P], float)
    f = lambda i: (100 * A[i, 0].sum() / A[i, 1].sum(), 100 * Bv[i, 0].sum() / Bv[i, 1].sum())
    t, e = f(np.arange(len(P)))
    bs = np.array([f(rng.integers(0, len(P), len(P))) for _ in range(2000)])
    gci = np.percentile(bs[:, 0] - bs[:, 1], [2.5, 97.5]); rci = np.percentile(bs[:, 0] / bs[:, 1], [2.5, 97.5])
    res[name] = dict(train=t, eval=e, gap=t - e, gap_ci=gci.tolist(), ratio=t / e, ratio_ci=rci.tolist(), bs=bs)
    so = self_open(key)
    print(f"{name:<11}{t:>7.1f}%{e:>7.1f}%{t-e:>+8.1f} [{gci[0]:+.1f},{gci[1]:+.1f}]{t/e:>7.2f}x [{rci[0]:.2f},{rci[1]:.2f}]"
          f"{'' if so is None else f'{so:>20.0f}%'}")

sft = [k for k in res if k.startswith("SFT")]
pool = np.concatenate([res[k]["bs"] for k in sft])
sft_ratio = np.mean([res[k]["ratio"] for k in sft]); sft_gap = np.mean([res[k]["gap"] for k in sft])
print(f"\nSFT-era mean: gap {sft_gap:+.1f} pts, ratio {sft_ratio:.2f}x "
      f"(range across steps {min(res[k]['ratio'] for k in sft):.2f}-{max(res[k]['ratio'] for k in sft):.2f}x)")
if "RLVR 2300" in res:
    r = res["RLVR 2300"]
    lo_sft = min(res[k]["ratio_ci"][0] for k in sft)
    verdict = ("SHRINKS" if r["ratio_ci"][1] < lo_sft else
               "PERSISTS (no clear shrink)" if r["ratio_ci"][0] > 1.2 else "UNCLEAR")
    print(f"PRIMARY  RLVR 2300 ratio {r['ratio']:.2f}x CI [{r['ratio_ci'][0]:.2f},{r['ratio_ci'][1]:.2f}] "
          f"vs lowest SFT-step CI bound {lo_sft:.2f}x  ->  {verdict}")
