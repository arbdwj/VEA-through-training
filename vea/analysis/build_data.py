"""Build data/ (corrected) from archive/v1/data/ (as published) + the template-fixed rerun labels.

Both trees use the stage layout data/<stage>/{vea,refusal}.jsonl. Replaced:
  sft/vea.jsonl        all 10 SFT steps re-sampled (+ the SFT merge, new)   <- paths.vea/sft_5e-5_*, sftmerge_*
  rlvr/vea.jsonl       re-sampled RLVR steps (50/1050/1800/2300 Ai2 template; 300/800 misaligned files)
  pretrain/vea.jsonl   Gemini-2.5-flash -> gpt-5-mini (outputs/pretrain_vea_gpt5mini.jsonl)
  */refusal.jsonl      re-sampled checkpoints -> paths.refusal (Fortress only; their stale
                       non-Fortress rows are dropped)
Everything else is copied unchanged, except v1-only artefacts that stay in the archive
(old/ Haiku labels; rlvr/vea_first_token.csv and vea_position*.csv, never part of the post). Template-gap runs (rlvrswap_*, rlvrtrain_*) are not data.
"""
import glob, json, os, shutil
from vea import config as C, paths
_cfg = C.settings()
ROOT = str(paths.ROOT)
SRC, DST = str(paths.ARCHIVE_V1), str(paths.DATA)
J, R = os.path.join(ROOT, _cfg["paths"]["vea"]), os.path.join(ROOT, _cfg["paths"]["refusal"])
REFUSAL_STAGE_DIR = {"sft": "sft", "sft_merge": "sft", "dpo": "dpo", "rlvr": "rlvr"}
rd = lambda p: [json.loads(l) for l in open(p)]


def wr(rel, rows):
    with open(os.path.join(DST, rel), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{rel}: {len(rows)} rows")


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    # v1-only artefacts (old/ judges, RLVR VEA-position CSVs) stay in the archive
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("old", "vea_*.csv"))

    new = {os.path.basename(p)[:-6]: rd(p) for p in glob.glob(os.path.join(J, "*.jsonl"))}
    new_sft = [r for k, v in new.items() if k.startswith("sft_5e-5") for r in v]
    new_merge = [r for k, v in new.items() if k.startswith("sftmerge") for r in v]
    new_rl = [r for k, v in new.items() if k.startswith("rlvr_step") for r in v]
    rl_steps = {r["step"] for r in new_rl}
    assert len(new_sft) == 10 * 3552 and len(new_rl) == sum(k.startswith("rlvr_step") for k in new) * 3552
    print(f"rerun RLVR steps replaced: {sorted(rl_steps)}")

    wr("sft/vea.jsonl", [r for r in rd(os.path.join(SRC, "sft/vea.jsonl")) if r["stage"] != "sft"] + new_sft + new_merge)
    kept = [r for r in rd(os.path.join(SRC, "rlvr/vea.jsonl")) if r["step"] not in rl_steps]
    is_f = lambda r: r["benchmark"] == "fortress"
    wr("rlvr/vea.jsonl", [r for r in kept if not is_f(r)] + [r for r in new_rl if not is_f(r)]
       + [r for r in kept if is_f(r)] + [r for r in new_rl if is_f(r)])
    wr("pretrain/vea.jsonl", rd(os.path.join(ROOT, "outputs/pretrain_vea_gpt5mini.jsonl")))

    replaced, ref_new = set(), []
    for p in glob.glob(os.path.join(R, "[!_]*.jsonl")):
        key = os.path.basename(p)[:-6]
        if key.startswith(("rlvrswap", "rlvrtrain")):
            continue
        vea = {(r["prompt_id"], r["rollout_id"]): r["vea"] for r in new[key]}
        for r in rd(p):
            r["vea"] = vea[(r["prompt_id"], r["rollout_id"])]
            ref_new.append(r); replaced.add((r["stage"], r["step"]))
    for d in ("sft", "dpo", "rlvr"):
        wr(f"{d}/refusal.jsonl", [r for r in rd(os.path.join(SRC, d, "refusal.jsonl")) if (r["stage"], r["step"]) not in replaced]
           + [r for r in ref_new if REFUSAL_STAGE_DIR[r["stage"]] == d])
