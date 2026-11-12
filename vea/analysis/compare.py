"""v1 (as published) vs corrected VEA rate for every re-sampled checkpoint."""
import collections
import glob
import json
import os

from vea import config as C, paths


def _rate(rows):
    ok = [r for r in rows if r.get("ok")]
    return (100 * sum(bool(r["vea"]) for r in ok) / len(ok), len(ok)) if ok else (float("nan"), 0)


def main():
    old = collections.defaultdict(list)
    for stage in ("sft", "dpo", "rlvr"):
        for r in map(json.loads, open(paths.ARCHIVE_V1 / stage / "vea.jsonl")):
            old[(r["stage"], r["step"])].append(r)
    new = {}
    for p in sorted(glob.glob(os.path.join(paths.ROOT, C.settings()["paths"]["vea"], "*.jsonl"))):
        rows = [json.loads(l) for l in open(p)]
        if rows:
            new[os.path.basename(p)[:-6]] = rows
    print(f"{'checkpoint':<22}{'v1 VEA':>9}{'new VEA':>9}{'change':>9}{'judged':>8}")
    print(f"{'dpo (v1 reference)':<22}{_rate(old[('dpo', 0)])[0]:>8.1f}%")
    for key, rows in sorted(new.items(), key=lambda kv: (kv[1][0]["stage"], kv[1][0]["step"])):
        o, _ = _rate(old.get((rows[0]["stage"], rows[0]["step"]), []))
        n, nn = _rate(rows)
        print(f"{key:<22}{o:>8.1f}%{n:>8.1f}%{n - o:>+9.1f}{nn:>8}")
