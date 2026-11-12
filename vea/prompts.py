"""The 444 evaluation prompts, in a fixed order (the order defines prompts_sha256)."""
import hashlib
import json

from vea import paths


def load(cfg):
    prompts = []
    for l in open(paths.ROOT / cfg["prompts"]["bench"]):
        r = json.loads(l)
        prompts.append({"prompt_id": r["prompt_id"], "benchmark": r["benchmark"], "prompt": r["prompt"]})
    for l in open(paths.ROOT / cfg["prompts"]["fortress"]):
        r = json.loads(l)
        p = {"prompt_id": f"fortress_{r['prompt_id']}", "benchmark": "fortress", "prompt": r["prompt"]}
        if "risk_domain" in r:
            p["risk_domain"] = r["risk_domain"]
        prompts.append(p)
    sha = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    return prompts, sha
