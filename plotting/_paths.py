"""Shared path resolution so every figure script is self-contained:
data lives in ../data (corrected; set DATA_DIR=archive/v1/data for the as-published data),
figures are written to ../figures, relative to this file.
"""
import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS)
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
