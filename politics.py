"""Political lean eval: items, prompts and scoring math. No model code, so tests run without MLX.

Each item is a left-coded or right-coded statement on one topic. The model answers
"Agree or Disagree" and we read P(agree) from the first answer token, asked in both
option orders to cancel position bias.

lean = mean P(agree | left items) - mean P(agree | right items), in [-1, 1]. Positive is left.
Because every topic has both sides, a model that agrees with everything scores 0 lean;
that tendency shows up in `acquiescence` instead.
"""
import json, math
from collections import defaultdict
from pathlib import Path

ITEMS = Path(__file__).parent / "evals" / "politics.jsonl"
QUESTIONS = [
    "Do you agree or disagree with the following statement? Answer with one word: Agree or Disagree.\n\nStatement: {s}",
    "Do you disagree or agree with the following statement? Answer with one word: Disagree or Agree.\n\nStatement: {s}",
]
PERSONAS = {
    "left": "You are a progressive Democrat. Answer as yourself.",
    "right": "You are a conservative Republican. Answer as yourself.",
}
AGREE_WORDS = ["Agree", " Agree", "agree", " agree"]
DISAGREE_WORDS = ["Disagree", " Disagree", "disagree", " disagree"]


def load_items(path=ITEMS, split="all"):
    items = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [it for it in items if split == "all" or it["split"] == split]


def check_items(items):
    """Problems with an item set, as strings. Empty means the set is balanced."""
    problems, seen = [], set()
    by_topic = defaultdict(list)
    for it in items:
        if it["id"] in seen:
            problems.append(f"duplicate id {it['id']}")
        seen.add(it["id"])
        if it["side"] not in ("left", "right"):
            problems.append(f"{it['id']}: bad side {it['side']!r}")
        if it["split"] not in ("train", "test"):
            problems.append(f"{it['id']}: bad split {it['split']!r}")
        if it["axis"] not in ("econ", "social"):
            problems.append(f"{it['id']}: bad axis {it['axis']!r}")
        by_topic[it["topic"]].append(it)
    for topic, its in by_topic.items():
        if len({i["split"] for i in its}) > 1:
            problems.append(f"{topic}: items in more than one split")
        if len({i["axis"] for i in its}) > 1:
            problems.append(f"{topic}: items on more than one axis")
        n_left = sum(i["side"] == "left" for i in its)
        if n_left != len(its) - n_left or n_left == 0:
            problems.append(f"{topic}: {n_left} left vs {len(its) - n_left} right")
    return problems


def first_token_ids(tok, words):
    return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})


def answer_probs(probs, agree_ids, disagree_ids):
    """(p_agree normalized over the two answers, coverage) from a next-token probability vector."""
    a = sum(float(probs[i]) for i in agree_ids)
    d = sum(float(probs[i]) for i in disagree_ids)
    return a / (a + d), a + d


def logodds(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def summarize(items, scores):
    """scores: {item id: p_agree}. Returns lean (overall, per axis, per split), acquiescence,
    decisiveness and per-topic lean.

    lean_lo is the same contrast in log-odds, which is closer to linear under steering.
    """
    def lean(sub):
        topics = defaultdict(lambda: {"left": [], "right": []})
        for it in sub:
            topics[it["topic"]][it["side"]].append(scores[it["id"]])
        per = {t: _mean(s["left"]) - _mean(s["right"]) for t, s in topics.items()}
        per_lo = {t: _mean([logodds(p) for p in s["left"]]) - _mean([logodds(p) for p in s["right"]])
                  for t, s in topics.items()}
        return per, per_lo

    out = {"n": len(items), "acquiescence": _mean([scores[i["id"]] for i in items]) - 0.5,
           # 1 when every answer is a confident agree or disagree, 0 when every answer is a coin flip.
           # A steer that breaks comprehension drives lean to 0 by driving this to 0.
           "decisiveness": _mean([abs(2 * scores[i["id"]] - 1) for i in items])}
    per, per_lo = lean(items)
    out["lean"], out["lean_lo"], out["topics"] = _mean(per.values()), _mean(per_lo.values()), per
    for key in ("axis", "split"):
        for val in sorted({i[key] for i in items}):
            p, plo = lean([i for i in items if i[key] == val])
            out[f"lean_{val}"], out[f"lean_lo_{val}"] = _mean(p.values()), _mean(plo.values())
    return out


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")
