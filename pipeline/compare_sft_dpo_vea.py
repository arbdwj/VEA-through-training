"""Compare VEA rates in the SFT target responses vs the DPO chosen/rejected responses,
restricted to the SAFETY / red-team subsets (where verbalized eval-awareness can occur;
the rest of these datasets is math/code/IF where VEA ~ 0).

  allenai/Dolci-Think-SFT-7B  -> WildJailbreak / WildGuardMix / CoCoNot  (target response)
  allenai/Dolci-Think-DPO-7B  -> wildjailbreak                            (chosen vs rejected)

Streams parquet shards (random order), keeps only safety-source examples until N collected,
judges VEA with the same Goodfire F1 rubric + extract_cot + Gemini-2.5-flash, shows tqdm
(kept/scanned while sampling; judged while judging), prints a comparison table. Resumable.

Run in your own shell:
    cd olmo_analysis && ./.venv/bin/python src/compare_sft_dpo_vea.py
    # overall (non-safety-filtered) instead:  SAFETY_ONLY=0 ./.venv/bin/python src/compare_sft_dpo_vea.py
    # tune:  N_SFT=2500 N_DPO=2500 JUDGE_WORKERS=20 ...
"""
import json
import os
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_vea as J

MODEL = os.environ.get("JUDGE_MODEL", "google/gemini-2.5-flash")
WORKERS = int(os.environ.get("JUDGE_WORKERS", "20"))
N_SFT = int(os.environ.get("N_SFT", "2500"))
N_DPO = int(os.environ.get("N_DPO", "2500"))          # DPO PAIRS
SAFETY_ONLY = os.environ.get("SAFETY_ONLY", "1") == "1"
SEED = 42
SFT_REPO = "allenai/Dolci-Think-SFT-7B"
DPO_REPO = "allenai/Dolci-Think-DPO-7B"
SFT_OUT = os.path.join(J.OUTPUTS, "sft_vea_safety.jsonl")
DPO_OUT = os.path.join(J.OUTPUTS, "dpo_vea_safety.jsonl")

SAFETY_SFT = ("wildjailbreak", "wildguard", "coconot")
SAFETY_DPO = ("wildjailbreak", "wildguard", "coconot")


def _shards(repo):
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.endswith(".parquet"))
    random.Random(SEED).shuffle(files)
    return files


def _cot(msg):
    return J.extract_cot(msg["content"] if isinstance(msg, dict) else str(msg))


def _is_safety(src, keys):
    s = (src or "").lower()
    return any(k in s for k in keys)


PEEK = int(os.environ.get("PEEK", "400"))   # abandon a shard if no safety row in its first PEEK rows


def sample(repo, n, keys, kind):
    """Stream shards (shuffled); within each shard, abandon early if no safety rows appear in
    the first PEEK rows (so non-safety shards are skipped cheaply instead of fully scanned)."""
    items, scanned, shards_used = [], 0, 0
    bar = tqdm(total=n, desc=f"sampling {kind}", ncols=90)
    for sh in _shards(repo):
        found = 0
        for k, ex in enumerate(load_dataset(repo, data_files=sh, split="train", streaming=True)):
            scanned += 1
            src = ex.get("dataset_source", "?")
            if not SAFETY_ONLY or _is_safety(src, keys):
                if kind == "SFT":
                    items.append({"cot": _cot(ex["messages"][-1]), "source": src})
                else:
                    items.append({"chosen": _cot(ex["chosen"][-1]), "rejected": _cot(ex["rejected"][-1]),
                                  "source": src, "chosen_model": ex.get("chosen_model"),
                                  "rejected_model": ex.get("rejected_model")})
                found += 1
                bar.update(1); bar.set_postfix(scanned=scanned, shards=shards_used)
            if len(items) >= n:
                break
            if SAFETY_ONLY and k + 1 >= PEEK and found == 0:
                break   # no safety in first PEEK rows of this shard -> skip the rest
        if found:
            shards_used += 1
        if len(items) >= n:
            break
    bar.close()
    print(f"  {kind}: kept {len(items)} (scanned {scanned}) | sources: "
          + ", ".join(f"{s.split('/')[-1][:24]}={c}" for s, c in
                      Counter(i['source'] for i in items).most_common(5)))
    return items


def _resume(path, key):
    done = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                done.add(key(r))
    return done


def judge(jobs, out, desc):
    client = J.make_client()
    lock = threading.Lock()
    fout = open(out, "a")

    def work(job):
        res = J.judge_one(client, job["cot"], "", MODEL)
        rec = {**{k: v for k, v in job.items() if k != "cot"},
               "ok": res["ok"], "vea": res["vea"], "judge": MODEL}
        with lock:
            fout.write(json.dumps(rec) + "\n"); fout.flush()
        return res["ok"]

    if jobs:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for _ in tqdm(as_completed([ex.submit(work, j) for j in jobs]),
                          total=len(jobs), desc=desc, ncols=90):
                pass
    fout.close()


def main():
    mode = "SAFETY subsets only" if SAFETY_ONLY else "OVERALL (no source filter)"
    print(f"=== VEA: SFT vs DPO | {mode} | judge={MODEL} ===\n")

    sft = sample(SFT_REPO, N_SFT, SAFETY_SFT, "SFT")
    done = _resume(SFT_OUT, lambda r: r["idx"])
    sj = [{"idx": i, "source": it["source"], "cot": it["cot"]}
          for i, it in enumerate(sft) if i not in done]
    print(f"[SFT] judging {len(sj)} (of {len(sft)}; {len(done)} cached)")
    judge(sj, SFT_OUT, "judging SFT")

    dpo = sample(DPO_REPO, N_DPO, SAFETY_DPO, "DPO")
    done = _resume(DPO_OUT, lambda r: (r["pair_id"], r["side"]))
    dj = []
    for i, it in enumerate(dpo):
        for side in ("chosen", "rejected"):
            if (i, side) not in done:
                dj.append({"pair_id": i, "side": side, "source": it["source"], "cot": it[side]})
    print(f"[DPO] judging {len(dj)} (of {2*len(dpo)}; {len(done)} cached)")
    judge(dj, DPO_OUT, "judging DPO")

    report()


def report():
    s = [r for r in (json.loads(l) for l in open(SFT_OUT))] if os.path.exists(SFT_OUT) else []
    s = [r for r in s if r.get("ok")]
    sv = sum(bool(r["vea"]) for r in s)
    pair = defaultdict(dict)
    if os.path.exists(DPO_OUT):
        for l in open(DPO_OUT):
            r = json.loads(l)
            if r.get("ok"):
                pair[r["pair_id"]][r["side"]] = bool(r["vea"])
    both = {p: v for p, v in pair.items() if "chosen" in v and "rejected" in v}
    n = len(both)
    cv = sum(v["chosen"] for v in both.values()); rv = sum(v["rejected"] for v in both.values())
    c_only = sum(1 for v in both.values() if v["chosen"] and not v["rejected"])
    r_only = sum(1 for v in both.values() if v["rejected"] and not v["chosen"])

    def p(k, t):
        return f"{100*k/t:.1f}%" if t else "-"

    print("\n" + "=" * 58)
    print(f"VEA rate — SFT vs DPO {'(safety subsets)' if SAFETY_ONLY else '(overall)'}  · Gemini-2.5-flash F1")
    print("=" * 58)
    print(f"{'dataset / role':<30}{'n':>7}{'VEA rate':>12}")
    print("-" * 58)
    print(f"{'SFT  target (DeepSeek-R1)':<30}{len(s):>7}{p(sv, len(s)):>12}")
    print(f"{'DPO  chosen   (qwen3-32b)':<30}{n:>7}{p(cv, n):>12}")
    print(f"{'DPO  rejected (qwen3-0.6b)':<30}{n:>7}{p(rv, n):>12}")
    print("=" * 58)
    if n:
        print("(DPO chosen>rejected is confounded by model size: 32b vs 0.6b.)")


if __name__ == "__main__":
    main()
