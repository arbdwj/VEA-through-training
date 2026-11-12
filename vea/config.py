"""Two kinds of config, plus infra settings:

  configs/models/<name>.yaml     WHAT to run: model repo, revision pattern, steps, output key,
                                 chat template, dataset, default sampling config
  configs/sampling/<name>.yaml   HOW to sample: n, temperature, top_p, seed, context, token budgets
  configs/settings.yaml          judge, local paths, Modal infra

A model config may `extends: <other>.yaml` (deep merge; e.g. rlvr_swap only changes the template).
`revision` and `key` are format strings over {step}. Configs can be given by name ("rlvr_swap")
or by path.
"""
from pathlib import Path

import yaml

from vea import paths

MODELS, SAMPLING = paths.CONFIGS / "models", paths.CONFIGS / "sampling"
REQUIRED_MODEL = ("stage", "model", "revision", "key", "steps", "template", "dataset", "sampling")
REQUIRED_SAMPLING = ("n", "temperature", "top_p", "seed", "max_model_len", "max_new_tokens",
                     "prompt_overflow", "min_new_tokens")


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _resolve(ref, folder):
    p = Path(ref)
    if p.is_file():
        return p
    p = folder / (ref if str(ref).endswith(".yaml") else f"{ref}.yaml")
    if not p.exists():
        raise FileNotFoundError(f"no config {ref!r} (looked in {folder})")
    return p


def _read(path):
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(_read(Path(path).parent / parent), cfg)
    return cfg


def _check(cfg, required, path):
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"{path}: missing {missing}")


def load_model(ref):
    path = _resolve(ref, MODELS)
    cfg = _read(path)
    _check(cfg, REQUIRED_MODEL, path)
    cfg["_name"] = path.stem
    return cfg


def load_sampling(ref):
    path = _resolve(ref, SAMPLING)
    cfg = _read(path)
    _check(cfg, REQUIRED_SAMPLING, path)
    cfg["_name"] = path.stem
    return cfg


def settings():
    return yaml.safe_load((paths.CONFIGS / "settings.yaml").read_text())


def parse_steps(steps, model_cfg):
    """'50,300' / [50, 300] / None or '' (-> every step in the model config)."""
    if steps in (None, "", []):
        return list(model_cfg["steps"])
    if isinstance(steps, (str, int)):
        return [int(s) for s in str(steps).split(",") if s.strip()]
    return [int(s) for s in steps]


def keys(model_cfg, steps=None):
    return [str(model_cfg["key"]).format(step=s) for s in parse_steps(steps, model_cfg)]


def jobs(model_ref, sampling_ref=None, steps=None):
    """Everything a generation container needs, one plain dict per checkpoint."""
    m = load_model(model_ref)
    s = load_sampling(sampling_ref or m["sampling"])
    st = settings()
    return [{
        "key": str(m["key"]).format(step=step),
        "stage": m["stage"],
        "lr": str(m.get("lr", "na")),
        "step": step,
        "model": m["model"],
        "revision": str(m["revision"]).format(step=step),
        "template": dict(m["template"]),
        "dataset": dict(m["dataset"]),
        "sampling": {k: (dict(v) if isinstance(v, dict) else v) for k, v in s.items() if k != "_name"},
        "model_config": m["_name"],
        "sampling_config": s["_name"],
        "out_dir": st["modal"]["out_dir"],
        "note": m.get("note", ""),
    } for step in parse_steps(steps, m)]
