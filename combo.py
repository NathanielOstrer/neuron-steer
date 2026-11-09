"""Pure helpers for the neuron-combination search: spec strings, the objective, batching,
gradient weights and candidate proposals. No model code, so tests and the GPU script share it."""
import math

KEYS = ("lean", "lean_lo", "lean_econ", "lean_social", "acquiescence", "decisiveness", "coverage")


def parse_spec(spec):
    """'9:7944:0.2,f9:7620:0.5' -> [('n', 9, 7944, 0.2), ('f', 9, 7620, 0.5)]."""
    out = []
    for item in filter(None, (s.strip() for s in (spec or "").split(","))):
        kind = "f" if item.startswith("f") else "n"
        parts = item.lstrip("f").split(":")
        if len(parts) != 3:
            raise ValueError(f"spec item {item!r} is not L:N:dose or fL:F:dose")
        out.append((kind, int(parts[0]), int(parts[1]), float(parts[2])))
    return out


def spec_str(items):
    """Inverse of parse_spec. Accepts (L, n, dose) tuples for neurons too."""
    parts = []
    for it in items:
        kind, L, n, d = it if len(it) == 4 else ("n", *it)
        parts.append(f"{'f' if kind == 'f' else ''}{L}:{n}:{d:g}")
    return ",".join(parts)


def objective(s, base, direction=-1.0, tol_acq=0.05, tol_dec=0.05, tol_cov=0.03, weight=4.0):
    """Higher is better. Rewards moving lean in `direction` (-1 is right) from the base lean.

    Penalizes the two ways a broken steer fakes a lean change: acquiescence moving more than
    tol_acq from base, and decisiveness falling more than tol_dec below base. Coverage falling
    more than tol_cov is penalized too, since then the model stops answering Agree/Disagree.
    """
    shift = direction * (s["lean"] - base["lean"])
    pen = (max(0.0, abs(s["acquiescence"] - base["acquiescence"]) - tol_acq)
           + max(0.0, base["decisiveness"] - s["decisiveness"] - tol_dec)
           + max(0.0, base["coverage"] - s["coverage"] - tol_cov))
    return shift - weight * pen


def pad_batch(seqs, pad_id=0):
    """Right-pad token lists. Returns (padded rows, lengths)."""
    T = max(len(s) for s in seqs)
    return [list(s) + [pad_id] * (T - len(s)) for s in seqs], [len(s) for s in seqs]


def length_batches(lengths, size):
    """Index groups of at most `size`, sorted by length so a batch wastes little padding."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i:i + size] for i in range(0, len(order), size)]


def grad_weights(p, sign, item_of, n_items):
    """Per-prompt weights that turn d(lo_i) into d(lean), d(acquiescence), d(decisiveness).

    p: P(agree) per prompt (sigmoid of its log-odds lo). sign: +1 for a left item, -1 for right.
    item_of: item index per prompt (each item is asked in two option orders).
    Uses the formulas in politics.summarize on a balanced set (3 left + 3 right per topic):
      lean = (2/N) sum_items sign * p_item,  acq = (1/N) sum p_item - 0.5,
      dec = (1/N) sum |2 p_item - 1|,        p_item = mean over that item's prompts.
    """
    per_item = {}
    for pi, it in zip(p, item_of):
        per_item.setdefault(it, []).append(pi)
    p_item = {it: sum(v) / len(v) for it, v in per_item.items()}
    w_lean, w_acq, w_dec = [], [], []
    for pi, s, it in zip(p, sign, item_of):
        k = len(per_item[it])
        dp = pi * (1 - pi) / k  # d p_item / d lo_i
        w_lean.append(2 * s * dp / n_items)
        w_acq.append(dp / n_items)
        w_dec.append(2 * math.copysign(1.0, 2 * p_item[it] - 1) * dp / n_items)
    return w_lean, w_acq, w_dec


def propose(pred, k, direction=-1.0, dose=0.2, acq_weight=1.0, dec_weight=1.0, exclude=()):
    """Rank neurons by predicted lean shift at `dose`, minus predicted acquiescence and
    decisiveness damage. pred: {'lean','acq','dec'} -> nested lists [layer][neuron] of the
    predicted change per unit dose. Each neuron gets the sign that moves lean in `direction`.
    Returns [(layer, neuron, signed dose, score)], best first."""
    rows = []
    for L, (rl, ra, rd) in enumerate(zip(pred["lean"], pred["acq"], pred["dec"])):
        for n, (dl, da, dd) in enumerate(zip(rl, ra, rd)):
            if (L, n) in exclude:
                continue
            sgn = 1.0 if direction * dl >= 0 else -1.0
            d = sgn * dose
            score = direction * dl * d - acq_weight * abs(da * d) + dec_weight * min(0.0, dd * d)
            rows.append((L, n, d, score))
    rows.sort(key=lambda r: -r[3])
    return rows[:k]


def omp(cols, target, k):
    """Orthogonal matching pursuit with numpy. cols: (d, m) unit columns. Returns
    [(column, coefficient)] and the cosine of the fit with the target after each step."""
    import numpy as np
    t = target / np.linalg.norm(target)
    chosen, cosines, r, coef = [], [], t.copy(), None
    for _ in range(k):
        scores = cols.T @ r
        scores[chosen] = 0
        chosen.append(int(np.argmax(np.abs(scores))))
        A = cols[:, chosen]
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        fit = A @ coef
        r = t - fit
        cosines.append(float(fit @ t / (np.linalg.norm(fit) + 1e-12)))
    return list(zip(chosen, [float(c) for c in coef])), cosines


def subset(items, frac):
    """Every other topic (frac 0.5) or all topics, so a quick screen keeps econ and social."""
    if frac >= 1:
        return items
    topics = sorted({i["topic"] for i in items})
    keep = set(topics[::round(1 / frac)])
    return [i for i in items if i["topic"] in keep]


def short(s):
    return {k: s[k] for k in KEYS}
