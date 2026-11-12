"""Build data/ (corrected): archive/v1/data/ with the template-fixed rerun (generation/gen_checkpoints.py) merged in.

  stage_vea.jsonl          SFT rows (all 10 steps) -> outputs/rerun_v2_judgments/sft_*
  rlvr_bench_vea.jsonl     re-sampled RLVR steps (50/1050/1800/2300 template; 300/800 misaligned files) -> rerun
  rlvr_fortress_vea.jsonl  steps 50/1050/1800/2300 -> rerun (Fortress rows)
  pretrain_vea_judgments   Gemini-2.5-flash -> gpt-5-mini (outputs/pretrain_vea_gpt5mini.jsonl)
  refusal_gpt5mini.jsonl   re-sampled checkpoints -> outputs/rerun_v2_refusal (Fortress only;
                           their stale non-Fortress rows are dropped)
Everything else is copied unchanged.
"""
import glob, json, os, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, DST = os.path.join(ROOT, "archive", "v1", "data"), os.path.join(ROOT, "data")
J, R = os.path.join(ROOT, "outputs/rerun_v2_judgments"), os.path.join(ROOT, "outputs/rerun_v2_refusal")
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("old"))
rd = lambda p: [json.loads(l) for l in open(p)]
def wr(name, rows):
    with open(os.path.join(DST, name), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{name}: {len(rows)} rows")

new = {os.path.basename(p)[:-6]: rd(p) for p in glob.glob(os.path.join(J, "*.jsonl"))}
new_sft = [r for k, v in new.items() if k.startswith("sft_5e-5") for r in v]
new_merge = [r for k, v in new.items() if k.startswith("sftmerge") for r in v]
new_rl = [r for k, v in new.items() if k.startswith("rlvr_step") for r in v]
rl_steps = {r["step"] for r in new_rl}
n_rl = sum(k.startswith("rlvr_step") for k in new)
assert len(new_sft) == 10 * 3552 and len(new_rl) == n_rl * 3552
print(f"rerun RLVR steps replaced: {sorted(rl_steps)}")

wr("stage_vea.jsonl", [r for r in rd(os.path.join(SRC, "stage_vea.jsonl")) if r["stage"] != "sft"] + new_sft + new_merge)
wr("rlvr_bench_vea.jsonl", [r for r in rd(os.path.join(SRC, "rlvr_bench_vea.jsonl")) if r["step"] not in rl_steps]
   + [r for r in new_rl if r["benchmark"] != "fortress"])
wr("rlvr_fortress_vea.jsonl", [r for r in rd(os.path.join(SRC, "rlvr_fortress_vea.jsonl")) if r["step"] not in rl_steps]
   + [r for r in new_rl if r["benchmark"] == "fortress"])
wr("pretrain_vea_judgments.jsonl", rd(os.path.join(ROOT, "outputs/pretrain_vea_gpt5mini.jsonl")))

replaced, ref_new = set(), []
for p in glob.glob(os.path.join(R, "[!_]*.jsonl")):
    key = os.path.basename(p)[:-6]
    if key.startswith(("rlvrswap", "rlvrtrain")):
        continue
    vea = {(r["prompt_id"], r["rollout_id"]): r["vea"] for r in new[key]}
    for r in rd(p):
        r["vea"] = vea[(r["prompt_id"], r["rollout_id"])]
        ref_new.append(r); replaced.add((r["stage"], r["step"]))
wr("refusal_gpt5mini.jsonl", [r for r in rd(os.path.join(SRC, "refusal_gpt5mini.jsonl"))
                              if (r["stage"], r["step"]) not in replaced] + ref_new)
