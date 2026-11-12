"""Two kinds of config, plus infra settings:

  configs/models/<name>.yaml     WHAT to run: pretrain, sft, dpo, rlvr. Every model config has
                                 exactly the same fields (MODEL_SCHEMA); the loader enforces it.
  configs/tmp_configs/<name>.yaml       ancillary experiments (same schema), e.g. the template-gap runs
  configs/sampling/<name>.yaml   HOW to sample: n, temperature, top_p, seed, context, token budgets
  configs/settings.yaml          judge, local paths, Modal infra

`revision` and `key` are format strings over {step}. A config is given by name ("rlvr",
"tmp_configs/rlvr_swap") or by path.
"""
from pathlib import Path

import yaml

from vea import paths

MODELS, SAMPLING = paths.CONFIGS / "models", paths.CONFIGS / "sampling"
MODEL_SCHEMA = {"stage", "lr", "model", "revision", "key", "steps", "template", "dataset", "sampling",
                "require_close_think", "note"}
TEMPLATE_SCHEMA = {"name", "repo", "rev", "system", "think", "newline"}
DATASET_SCHEMA = {"bench", "fortress"}
REQUIRED_SAMPLING = ("n", "temperature", "top_p", "seed", "max_model_len", "max_new_tokens",
                     "prompt_overflow", "min_new_tokens")


def _resolve(ref, folder):
    p = Path(ref)
    if p.is_file():
        return p
    name = ref if str(ref).endswith(".yaml") else f"{ref}.yaml"
    for base in (folder, paths.CONFIGS):          # "rlvr" -> models/rlvr.yaml; "tmp_configs/rlvr_swap" -> configs/tmp_configs/...
        if (base / name).is_file():
            return base / name
    raise FileNotFoundError(f"no config {ref!r} (looked in {folder} and {paths.CONFIGS})")


def _read(path):
    return yaml.safe_load(Path(path).read_text()) or {}


def _check(d, schema, where, exact=True):
    missing, extra = schema - set(d), set(d) - schema
    if missing or (exact and extra):
        raise KeyError(f"{where}: " + (f"missing {sorted(missing)} " if missing else "")
                       + (f"unexpected {sorted(extra)}" if exact and extra else ""))


def load_model(ref):
    path = _resolve(ref, MODELS)
    cfg = _read(path)
    _check(cfg, MODEL_SCHEMA, path)
    _check(cfg["template"], TEMPLATE_SCHEMA, f"{path} template")
    _check(cfg["dataset"], DATASET_SCHEMA, f"{path} dataset")
    cfg["_name"] = path.stem
    return cfg


def load_sampling(ref):
    path = _resolve(ref, SAMPLING)
    cfg = _read(path)
    _check(cfg, set(REQUIRED_SAMPLING), path)
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
