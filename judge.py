"""Judge generated rollouts (gpt-5-mini; settings under `judge:` in the config).

  python judge.py vea --model rlvr_swap                  # every step in configs/models/rlvr_swap.yaml
  python judge.py vea --model rlvr_swap --steps 50,800
  python judge.py vea --keys sft_5e-5_step10000,rlvr_step_0050
  python judge.py vea                                    # every finished generation job
  python judge.py refusal --model rlvr                   # refusal (Fortress) + joined VEA
  python judge.py refusal --calibrate                    # rubric vs v1 labels on unchanged DPO
  python judge.py vea --input 'data/pretrain_rollouts/*.jsonl' \\
        --output outputs/pretrain_vea_gpt5mini.jsonl --model pretrain    # arbitrary rollout files

Judge model / effort / workers and output paths: configs/settings.yaml.

Raw rollouts are pulled from the Modal outputs volume into paths.raw if missing. Labels go to
paths.vea/<key>.jsonl and paths.refusal/<key>.jsonl. Resumable: re-run to finish or retry.
"""
import argparse
import glob
import json
import os

from vea import config as C, judging as JG, paths, rubrics as R

ROOT = str(paths.ROOT)
VEA_FIELDS = ("stage", "lr", "step", "benchmark", "prompt_id", "rollout_id", "n_tokens", "finish_reason",
              "revision", "template", "template_rev")


def _rows(path):
    return [json.loads(l) for l in open(path)]


def _keys(args):
    if args.keys:
        return [k.strip() for k in args.keys.split(",") if k.strip()]
    if args.model:
        return C.keys(C.load_model(args.model), args.steps)
    return None


def vea(args, cfg):
    jcfg = cfg["judge"]
    if args.input:                                     # arbitrary rollout files -> one output file
        rows = [r for p in sorted(glob.glob(os.path.join(ROOT, args.input))) for r in _rows(p)]
        tasks = {os.path.join(ROOT, args.output): rows}
    else:
        keys = _keys(args) or JG.finished_keys(cfg)
        tasks = {os.path.join(ROOT, cfg["paths"]["vea"], f"{k}.jsonl"): _rows(JG.pull(k, cfg, ROOT)) for k in keys}
    if args.model and C.load_model(args.model).get("judge_require_close_think"):
        # no CoT to judge -> VEA=False by construction, not written
        tasks = {p: [r for r in rows if "</think>" in r["completion"]] for p, rows in tasks.items()}

    def make_row(r, res):
        row = {k: r[k] for k in VEA_FIELDS if k in r}
        row.update(ok=True, vea=res["vea"], quote=res["quote"], judge=jcfg["model"],
                   reasoning_effort=jcfg["reasoning_effort"])
        return row

    JG.run(tasks, lambda cl, r: JG.vea_one(cl, jcfg, r["completion"]), make_row, jcfg, "VEA")


def _refusal_row(jcfg, extra):
    def make_row(r, res):
        row = {k: r.get(k) for k in ("stage", "benchmark", "step", "prompt_id", "rollout_id")}
        row.update(extra(r), ok=True, refusal=res["refusal"], no_answer=res["no_answer"], judge=jcfg["model"],
                   reasoning_effort=jcfg["reasoning_effort"], rubric=R.REFUSAL_RUBRIC)
        return row
    return make_row


def refusal(args, cfg):
    jcfg, benches = cfg["judge"], set(cfg["judge"]["refusal_benchmarks"])
    judge_fn = lambda cl, r: JG.refusal_one(cl, jcfg, r["completion"], r["prompt"])
    out_dir = os.path.join(ROOT, cfg["paths"]["refusal"])
    if args.calibrate:
        calibrate(cfg, jcfg, benches, judge_fn, out_dir)
        return
    keys = _keys(args)
    raw_dir = os.path.join(ROOT, cfg["paths"]["raw"])
    if keys is None:                                   # every local raw file
        keys = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(raw_dir, "*.jsonl")))
    tasks = {}
    for k in keys:
        vp = os.path.join(ROOT, cfg["paths"]["vea"], f"{k}.jsonl")
        vlab = {(r["prompt_id"], r["rollout_id"]): r["vea"] for r in _rows(vp)} if os.path.exists(vp) else {}
        rows = [r for r in _rows(JG.pull(k, cfg, ROOT)) if r["benchmark"] in benches]
        for r in rows:                                 # join this rollout's VEA label onto its refusal row
            r["_vea"] = vlab.get((r["prompt_id"], r["rollout_id"]))
        tasks[os.path.join(out_dir, f"{k}.jsonl")] = rows
    make = _refusal_row(jcfg, lambda r: {"vea": r["_vea"], "template_rev": r.get("template_rev")})
    JG.run(tasks, judge_fn, make, jcfg, "refusal")


def calibrate(cfg, jcfg, benches, judge_fn, out_dir):
    """Agreement of the refusal rubric with the v1 labels on DPO main (rollouts unchanged by the fix)."""
    old_raw = os.path.join(ROOT, "outputs", "old_raw")
    src = os.path.join(old_raw, "stage_rollouts_dpo_main.jsonl")
    if not os.path.exists(src):
        os.makedirs(old_raw, exist_ok=True)
        import subprocess
        subprocess.run(["modal", "volume", "get", "--force", cfg["modal"]["outputs_volume"],
                        "stage_rollouts_dpo_main.jsonl", old_raw + "/"], check=True, stdout=subprocess.DEVNULL)
    out = os.path.join(out_dir, "_calib_dpo.jsonl")
    JG.run({out: [r for r in _rows(src) if r["benchmark"] in benches]}, judge_fn, _refusal_row(jcfg, lambda r: {}),
           jcfg, "calibrate")
    old = {(r["prompt_id"], r["rollout_id"]): r for r in _rows(paths.ARCHIVE_V1 / "refusal_gpt5mini.jsonl")
           if r["stage"] == "dpo" and r["benchmark"] in benches}
    new = {(r["prompt_id"], r["rollout_id"]): r for r in _rows(out)}
    pairs = [(old[k]["refusal"], new[k]["refusal"]) for k in new
             if k in old and old[k].get("refusal") is not None and new[k]["refusal"] is not None]
    agree = sum(a == b for a, b in pairs)
    print(f"DPO calibration: n={len(pairs)} agreement={100 * agree / len(pairs):.1f}%  "
          f"v1 refusal={100 * sum(a for a, _ in pairs) / len(pairs):.1f}%  "
          f"new={100 * sum(b for _, b in pairs) / len(pairs):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["vea", "refusal"])
    ap.add_argument("--model", help="model config name/path: which checkpoints (keys) to judge")
    ap.add_argument("--steps", default="", help="subset of the model config's steps, e.g. 50,800")
    ap.add_argument("--keys", default="", help="explicit checkpoint keys instead of --model")
    ap.add_argument("--input", help="glob of rollout files (vea only); requires --output")
    ap.add_argument("--output")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--calibrate", action="store_true", help="refusal only")
    a = ap.parse_args()
    cfg = C.settings()
    if a.workers:
        cfg["judge"]["workers"] = a.workers
    if a.input and not a.output:
        ap.error("--input needs --output")
    vea(a, cfg) if a.what == "vea" else refusal(a, cfg)
