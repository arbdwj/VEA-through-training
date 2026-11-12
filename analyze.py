"""Analyses and figures over the judged data.

  python analyze.py build-data        # archive/v1/data + rerun labels -> data/ (corrected)
  python analyze.py compare           # v1 vs corrected VEA per re-sampled checkpoint
  python analyze.py template-gap      # training-vs-eval template gap, SFT -> RLVR (+ one-token swap)
  python analyze.py eval-gaming       # refusal ~ VEA x RLVR step (conditional logit)
  python analyze.py pretrain-judges   # pretraining VEA: Gemini (v1) vs gpt-5-mini
  python analyze.py sanity            # configs/sanity/*: re-run checkpoints vs existing data
  python analyze.py figures [fig1 fig_template_gap ...] [--data-dir archive/v1/data] [--fig-dir DIR]
"""
import argparse
import glob
import importlib
import os
import subprocess
import sys

from vea import paths

ANALYSES = {"build-data": "build_data", "compare": "compare", "template-gap": "template_gap",
            "eval-gaming": "eval_gaming", "pretrain-judges": "pretrain_judges", "sanity": "sanity"}
FIG_DIR = paths.ROOT / "vea" / "figures"


def figures(names, data_dir, fig_dir):
    scripts = sorted(glob.glob(str(FIG_DIR / "fig*.py")),
                     key=lambda p: (not os.path.basename(p)[3].isdigit(), int("".join(c for c in os.path.basename(p)[3:6] if c.isdigit()) or 0), p))
    if names:
        scripts = [s for s in scripts if any(os.path.basename(s)[:-3] == n or os.path.basename(s).startswith(n + "_") for n in names)]
        if not scripts:
            sys.exit(f"no figure script matches {names}")
    env = dict(os.environ, FIG1_YMAX=os.environ.get("FIG1_YMAX", "3.2"))   # same y-range as the published Fig 1
    if data_dir:
        env["DATA_DIR"] = str(paths.ROOT / data_dir)
    if fig_dir:
        env["FIG_DIR"] = str(paths.ROOT / fig_dir)
    for s in scripts:
        r = subprocess.run([sys.executable, s], env=env, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"  {'ok ' if r.returncode == 0 else 'ERR'} {os.path.basename(s):<40} {tail[:110]}")
        if r.returncode:
            print(r.stderr[-1500:])
    print(f"-> {fig_dir or paths.FIGURES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=[*ANALYSES, "figures"])
    ap.add_argument("names", nargs="*", help="figures: subset, e.g. fig1 fig_template_gap")
    ap.add_argument("--data-dir", help="figures: read this data dir instead of data/ (e.g. archive/v1/data)")
    ap.add_argument("--fig-dir", help="figures: write here instead of figures/")
    a = ap.parse_args()
    if a.what == "figures":
        figures(a.names, a.data_dir, a.fig_dir)
    else:
        importlib.import_module(f"vea.analysis.{ANALYSES[a.what]}").main()
