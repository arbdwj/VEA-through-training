"""Shared path resolution so every figure script is self-contained:
data lives in ../data, figures are written to ../figures, relative to this file.
"""
import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS)
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def d(*parts):
    return os.path.join(DATA, *parts)


def f(name):
    return os.path.join(FIG, name)
