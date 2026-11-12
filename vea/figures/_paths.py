"""Shared path resolution so every figure script is self-contained:
data lives in <repo>/data (corrected; set DATA_DIR=archive/v1/data for the as-published data),
figures are written to <repo>/figures, relative to this file.
"""
import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPTS))
# DATA can be redirected with DATA_DIR (e.g. archive/v1/data = as published)
DATA = os.environ.get("DATA_DIR") or os.path.join(ROOT, "data")
if not os.path.isabs(DATA):
    DATA = os.path.join(ROOT, DATA)
# FIG can be redirected with FIG_DIR (e.g. to regenerate into a parallel folder)
FIG = os.environ.get("FIG_DIR") or os.path.join(ROOT, "figures")
if not os.path.isabs(FIG):
    FIG = os.path.join(ROOT, FIG)
os.makedirs(FIG, exist_ok=True)


def d(*parts):
    return os.path.join(DATA, *parts)


def f(name):
    return os.path.join(FIG, name)


STAGES = ("pretrain", "base", "sft", "dpo", "rlvr")


def rows(name, *stages):
    """Concatenate <DATA>/<stage>/<name> (JSONL) over the given stages (default: all that have it)."""
    import json
    out = []
    for s in stages or STAGES:
        p = d(s, name)
        if os.path.exists(p):
            out += [json.loads(l) for l in open(p)]
    return out


def refusal_rows():
    """Refusal labels for every stage; REFUSAL_FILE (relative to DATA) overrides, e.g. old/refusal_haiku.jsonl."""
    import json
    alt = os.environ.get("REFUSAL_FILE")
    return [json.loads(l) for l in open(d(alt))] if alt else rows("refusal.jsonl")
