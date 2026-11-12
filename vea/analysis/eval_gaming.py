"""Pre-specified test of "eval-gaming increases over RLVR" (written before the corrected
step-300/800 data was judged).

Data: Fortress rollouts at the 10 RLVR steps, template-fixed (data/ + outputs/rerun_v2_refusal).
Model: conditional logistic regression, strata = question x step (so every comparison is
within the same question at the same checkpoint):
    refusal ~ VEA + VEA:step_k        (step_k = step / 1000)
PRIMARY: coefficient on VEA:step_k > 0 with 95% CI (question-cluster bootstrap) excluding 0.
SECONDARY: VEA log-odds ratio fitted separately on early (50-800) vs late (1050-2300) RLVR.
Also reports the old within-question gap (>=2 VEA & >=2 non-VEA) for reference.
"""
import collections, glob, json, os, sys
import numpy as np
from statsmodels.discrete.conditional_models import ConditionalLogit

from vea import config as C, paths
_cfg = C.settings()
ROOT = str(paths.ROOT)
STEPS = [50, 300, 550, 800, 1050, 1300, 1550, 1800, 2050, 2300]


def load():
    rows = {}
    ok = lambda r: r.get("ok") and not r.get("no_answer") and r.get("refusal") is not None and r.get("vea") is not None
    for r in map(json.loads, open(os.path.join(ROOT, "data/rlvr/refusal.jsonl"))):
        if r["stage"] == "rlvr" and r["benchmark"] == "fortress" and ok(r):
            rows[(r["step"], r["prompt_id"], r["rollout_id"])] = r
    for s in STEPS:  # rerun files override (includes 300/800 re-sampled)
        p = os.path.join(ROOT, _cfg["paths"]["refusal"], f"rlvr_step_{s:04d}.jsonl")
        vp = os.path.join(ROOT, _cfg["paths"]["vea"], f"rlvr_step_{s:04d}.jsonl")
        if not os.path.exists(p):
            continue
        vea = {(r["prompt_id"], r["rollout_id"]): r["vea"] for r in map(json.loads, open(vp))}
        for k in [k for k in rows if k[0] == s]:
            del rows[k]
        for r in map(json.loads, open(p)):
            r["vea"] = vea.get((r["prompt_id"], r["rollout_id"]))
            if ok(r):
                rows[(s, r["prompt_id"], r["rollout_id"])] = r
    return list(rows.values())


def design(R, steps):
    R = [r for r in R if r["step"] in steps]
    g = collections.defaultdict(list)
    for r in R:
        g[(r["prompt_id"], r["step"])].append(r)
    keep = [r for v in g.values() if len({x["refusal"] for x in v}) > 1 and len({x["vea"] for x in v}) > 1 for r in v]
    y = np.array([float(r["refusal"]) for r in keep])
    v = np.array([float(r["vea"]) for r in keep]); st = np.array([r["step"] / 1000 for r in keep])
    grp = np.array([hash((r["prompt_id"], r["step"])) for r in keep])
    return y, v, st, grp, keep


def fit(R, steps, interaction):
    y, v, st, grp, keep = design(R, steps)
    if len(y) < 20:
        return None, len({(r['prompt_id'], r['step']) for r in keep})
    X = np.column_stack([v, v * st]) if interaction else v[:, None]
    m = ConditionalLogit(y, X, groups=grp).fit(disp=0)
    return m.params, len({(r['prompt_id'], r['step']) for r in keep})


def boot(R, steps, interaction, idx, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    qs = sorted({r["prompt_id"] for r in R}, key=str)
    byq = collections.defaultdict(list)
    for r in R:
        byq[r["prompt_id"]].append(r)
    out = []
    for b in range(B):
        samp = []
        for j, q in enumerate(rng.choice(len(qs), len(qs))):
            for r in byq[qs[q]]:
                rr = dict(r); rr["prompt_id"] = f"{qs[q]}#{j}"; samp.append(rr)
        try:
            p, _ = fit(samp, steps, interaction)
            if p is not None:
                out.append(p[idx])
        except Exception:
            pass
    return np.percentile(out, [2.5, 97.5]), len(out)


def main():
    R = load()
    have = sorted({r["step"] for r in R})
    print(f"RLVR Fortress rollouts: {len(R)} | steps: {have}")
    p, n = fit(R, STEPS, True)
    ci, nb = boot(R, STEPS, True, 1)
    print(f"\nPRIMARY  VEA x step interaction: {p[1]:+.3f} log-odds per 1000 steps  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
          f"  (strata used: {n}, bootstrap fits: {nb})")
    print(f"         VEA main effect at step 0: {p[0]:+.3f} log-odds")
    for name, ss in (("early 50-800", STEPS[:4]), ("late 1050-2300", STEPS[4:])):
        q, n = fit(R, ss, False)
        c, _ = boot(R, ss, False, 0, B=500)
        print(f"SECONDARY {name:<15} VEA log-OR {q[0]:+.2f} (OR {np.exp(q[0]):.1f})  95% CI [{c[0]:+.2f}, {c[1]:+.2f}]  strata {n}")


if __name__ == "__main__":
    main()
