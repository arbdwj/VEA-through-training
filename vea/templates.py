"""Chat-template handling: never trust the checkpoint branch's template.

load_tokenizer() takes the checkpoint's tokenizer (vocab) and overwrites its chat_template
with the one pinned in the config. render() renders every prompt and asserts the exact
string (and, for the one-token swap, the exact token change) before any GPU time is spent.
"""


def expected_text(spec, prompt):
    sys_turn = f"<|im_start|>system\n{spec['system']}<|im_end|>\n" if spec.get("system") else ""
    text = f"{sys_turn}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if spec.get("think"):
        text += "<think>"
    if spec.get("newline"):
        text += "\n"
    return text


def load_tokenizer(model, revision, spec, cache_dir=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, revision=revision, cache_dir=cache_dir)
    tmpl = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["rev"], cache_dir=cache_dir)
    tok.chat_template = tmpl.chat_template
    return tok


def render(tok, prompts, spec):
    """-> list of (text, ids). Raises on any deviation from expected_text()."""
    out = []
    gt = tok(">", add_special_tokens=False)["input_ids"]
    nl = tok(">\n", add_special_tokens=False)["input_ids"]
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                       add_generation_prompt=True, tokenize=False)
        base_ids = None
        if spec.get("newline"):                      # one-token swap on top of the template
            base_ids = tok(text, add_special_tokens=False)["input_ids"]
            text += "\n"
        expected = expected_text(spec, p["prompt"])
        if text != expected:
            raise AssertionError(f"rendered prompt mismatch for {p['prompt_id']}:\n{text[:300]!r}\n"
                                 f"--- expected ---\n{expected[:300]!r}")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if base_ids is not None and not (len(nl) == 1 and len(gt) == 1 and base_ids[-1] == gt[0]
                                         and ids[-1] == nl[0] and ids[:-1] == base_ids[:-1]):
            raise AssertionError(f"one-token swap failed for {p['prompt_id']}: "
                                 f"{tok.convert_ids_to_tokens(base_ids[-2:])} -> {tok.convert_ids_to_tokens(ids[-2:])}")
        if tok.decode(ids) != text:
            raise AssertionError(f"token round-trip mismatch for {p['prompt_id']}")
        out.append((text, ids))
    return out


def budgets(prompts, rendered, sampling):
    """Per-prompt (ids, max_new_tokens), applying the overflow policy."""
    mml, floor = sampling["max_model_len"], sampling["min_new_tokens"]
    caps = sampling["max_new_tokens"]
    out = []
    for p, (_, ids) in zip(prompts, rendered):
        if mml - len(ids) < floor:
            if sampling["prompt_overflow"] != "truncate":
                raise AssertionError(f"{p['prompt_id']}: {len(ids)} prompt tokens leave <{floor} of {mml} context "
                                     f"(set sampling.prompt_overflow: truncate to allow)")
            ids = ids[:mml - floor]
        cap = min(caps.get(p["benchmark"], caps["default"]), mml - len(ids))
        out.append((ids, cap))
    return out
