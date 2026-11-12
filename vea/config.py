"""YAML experiment configs -> per-checkpoint job dicts.

A config may `extends: <other>.yaml` (resolved relative to its own directory); the child is
deep-merged over the parent. `revision` and `key` are format strings over {step}.
"""
from pathlib import Path

import yaml

from vea import paths


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = paths.CONFIGS / path
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(load(path.parent / parent), cfg)
    cfg["_path"] = str(path)
    return cfg


def base():
    return load(paths.CONFIGS / "base.yaml")


def parse_steps(steps, cfg):
    """'50,300' / [50, 300] / None (-> all steps in the config)."""
    if steps in (None, "", []):
        return list(cfg["steps"])
    if isinstance(steps, str):
        return [int(s) for s in steps.split(",") if s.strip()]
    return [int(s) for s in steps]


def job(cfg, step):
    """Everything a generation container needs for one checkpoint (plain, picklable dict)."""
    for field in ("stage", "model", "revision", "key"):
        if field not in cfg:
            raise KeyError(f"{cfg['_path']}: missing '{field}'")
    tname = cfg.get("template", "eval")
    tmpl = dict(cfg["templates"][tname], name=tname)
    s = cfg["sampling"]
    return {
        "key": str(cfg["key"]).format(step=step),
        "stage": cfg["stage"],
        "lr": str(cfg.get("lr", "na")),
        "step": step,
        "model": cfg["model"],
        "revision": str(cfg["revision"]).format(step=step),
        "template": tmpl,
        "sampling": dict(s, max_new_tokens=dict(s["max_new_tokens"])),
        "out_dir": cfg["modal"]["out_dir"],
        "progress_dict": cfg["modal"]["progress_dict"],
        "note": cfg.get("note", ""),
        "config": Path(cfg["_path"]).name,
    }


def jobs(cfg, steps=None):
    return [job(cfg, s) for s in parse_steps(steps, cfg)]
