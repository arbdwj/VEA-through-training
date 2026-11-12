"""Repo-level paths (everything else comes from configs/*.yaml)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"                       # corrected data (v2)
ARCHIVE_V1 = ROOT / "archive" / "v1" / "data"   # as published
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
