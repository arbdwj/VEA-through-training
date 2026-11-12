"""By-prompt cluster bootstrap for clustered (rollouts nested in prompts) rates.

VEA/refusal rollouts are not independent: 8 rollouts share a prompt, and the outcome is
strongly prompt-driven (ICC ~0.45). A naive Wald CI on N rollouts is ~2x too narrow. The
correct estimator resamples PROMPTS with replacement and recomputes the pooled rate.
"""
import numpy as np

B_DEFAULT = 5000


def cluster_ci(groups, B=B_DEFAULT, seed=0):
    """groups: dict prompt_id -> iterable of 0/1 outcomes (one per rollout of that prompt).

    Returns (rate%, yerr_low%, yerr_high%) for an asymmetric 95% percentile-bootstrap CI,
    where yerr_low/high are the distances from the point estimate (ready for plt.errorbar).
    """
    keys = [k for k in groups if len(groups[k]) > 0]
    if not keys:
        return 0.0, 0.0, 0.0
    arrs = [np.asarray(groups[k], dtype=float) for k in keys]
    allv = np.concatenate(arrs)
    point = 100.0 * allv.mean()
    n = len(keys)
    rng = np.random.default_rng(seed)
    ests = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        ests[b] = np.concatenate([arrs[i] for i in idx]).mean()
    lo, hi = np.percentile(ests, [2.5, 97.5]) * 100.0
    return point, max(point - lo, 0.0), max(hi - point, 0.0)
