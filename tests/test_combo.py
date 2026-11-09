"""combo.py: spec strings, the objective, batching, gradient weights, proposals and OMP. No model."""
import math
import random
import numpy as np
import pytest

import combo
import politics


def test_parse_spec_reads_neurons_and_features():
    assert combo.parse_spec("9:7944:0.2, f9:7620:-0.5") == [("n", 9, 7944, 0.2), ("f", 9, 7620, -0.5)]


def test_parse_spec_empty_is_no_steer():
    assert combo.parse_spec("") == [] and combo.parse_spec(None) == []


def test_parse_spec_rejects_malformed_item():
    with pytest.raises(ValueError):
        combo.parse_spec("9:7944")


def test_spec_str_round_trips_and_accepts_plain_triples():
    s = "9:7944:0.2,f9:7620:-0.5,5:6096:-0.15"
    assert combo.spec_str(combo.parse_spec(s)) == s
    assert combo.spec_str([(9, 7944, 0.2), (5, 1, -0.1)]) == "9:7944:0.2,5:1:-0.1"


BASE = {"lean": 0.40, "acquiescence": -0.05, "decisiveness": 0.50, "coverage": 0.98}


def summ(**kw):
    return {**BASE, **kw}


def test_objective_rewards_a_rightward_shift_with_no_side_effects():
    assert combo.objective(summ(lean=0.10), BASE) == pytest.approx(0.30)


def test_objective_is_zero_at_base_and_negative_for_a_leftward_shift():
    assert combo.objective(BASE, BASE) == 0
    assert combo.objective(summ(lean=0.5), BASE) == pytest.approx(-0.1)


def test_objective_direction_plus_one_rewards_left():
    assert combo.objective(summ(lean=0.5), BASE, direction=1.0) == pytest.approx(0.1)


def test_objective_ignores_changes_inside_tolerance():
    s = summ(lean=0.1, acquiescence=-0.01, decisiveness=0.46, coverage=0.96)
    assert combo.objective(s, BASE) == pytest.approx(0.30)


@pytest.mark.parametrize("field,value,excess", [
    ("acquiescence", 0.10, 0.10),    # +0.15 from base, 0.05 tolerated
    ("acquiescence", -0.20, 0.10),   # either direction counts
    ("decisiveness", 0.30, 0.15),    # fell 0.20, 0.05 tolerated
    ("coverage", 0.90, 0.05),        # fell 0.08, 0.03 tolerated
])
def test_objective_penalizes_each_fake_lean_route(field, value, excess):
    s = summ(lean=0.1, **{field: value})
    assert combo.objective(s, BASE) == pytest.approx(0.30 - 4.0 * excess)


def test_objective_does_not_penalize_more_decisiveness():
    assert combo.objective(summ(lean=0.1, decisiveness=0.9), BASE) == pytest.approx(0.30)


def test_collapse_to_zero_lean_scores_worse_than_doing_nothing():
    broken = summ(lean=0.0, decisiveness=0.05)
    assert combo.objective(broken, BASE) < 0


def test_pad_batch_right_pads_and_reports_lengths():
    rows, lens = combo.pad_batch([[1, 2, 3], [4]], pad_id=0)
    assert rows == [[1, 2, 3], [4, 0, 0]] and lens == [3, 1]


def test_length_batches_cover_every_index_once_in_length_order():
    lens = [5, 1, 9, 3, 7, 2, 8]
    groups = combo.length_batches(lens, 3)
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(len(lens)))
    assert [lens[i] for i in flat] == sorted(lens)
    assert all(len(g) <= 3 for g in groups)


def _items(n_topics=2):
    items = []
    for t in range(n_topics):
        for side in ("left", "right"):
            for k in range(3):
                items.append({"id": f"{t}-{side}-{k}", "topic": f"t{t}", "side": side, "axis": "econ", "split": "train"})
    return items


def _summary_from_lo(items, lo):
    """politics.summarize on p = sigmoid(lo), with two prompts per item."""
    scores = {}
    for i, it in enumerate(items):
        ps = [1 / (1 + math.exp(-x)) for x in lo[2 * i:2 * i + 2]]
        scores[it["id"]] = sum(ps) / 2
    return politics.summarize(items, scores)


def test_grad_weights_match_finite_differences_of_the_summary():
    rng = random.Random(1)
    items = _items()
    lo = [rng.uniform(-3, 3) for _ in range(2 * len(items))]
    item_of = [i for i in range(len(items)) for _ in range(2)]
    sign = [1.0 if items[i]["side"] == "left" else -1.0 for i in item_of]
    p = [1 / (1 + math.exp(-x)) for x in lo]
    wl, wa, wd = combo.grad_weights(p, sign, item_of, len(items))
    # Move every prompt's log-odds along a random direction v; the summary moves by w . v.
    v = [rng.uniform(-1, 1) for _ in lo]
    eps = 1e-5
    hi = _summary_from_lo(items, [x + eps * d for x, d in zip(lo, v)])
    low = _summary_from_lo(items, [x - eps * d for x, d in zip(lo, v)])
    for key, w in (("lean", wl), ("acquiescence", wa), ("decisiveness", wd)):
        fd = (hi[key] - low[key]) / (2 * eps)
        assert sum(a * b for a, b in zip(w, v)) == pytest.approx(fd, rel=1e-4, abs=1e-8), key


def test_grad_weights_on_left_and_right_items_have_opposite_lean_sign():
    items = _items(1)
    item_of = [i for i in range(len(items)) for _ in range(2)]
    sign = [1.0 if items[i]["side"] == "left" else -1.0 for i in item_of]
    wl, wa, _ = combo.grad_weights([0.5] * len(item_of), sign, item_of, len(items))
    assert all((w > 0) == (s > 0) for w, s in zip(wl, sign))
    assert all(w > 0 for w in wa)


def test_propose_picks_the_sign_that_moves_lean_right():
    pred = {"lean": [[-1.0, 2.0]], "acq": [[0.0, 0.0]], "dec": [[0.0, 0.0]]}
    rows = combo.propose(pred, 2, dose=0.1)
    assert {(r[1], r[2]) for r in rows} == {(0, 0.1), (1, -0.1)}
    assert rows[0][1] == 1  # larger predicted shift first


def test_propose_penalizes_acquiescence_and_lost_decisiveness():
    pred = {"lean": [[-1.0, -1.0, -1.0]], "acq": [[0.0, 5.0, 0.0]], "dec": [[0.0, 0.0, -5.0]]}
    rows = combo.propose(pred, 3, dose=0.1)
    assert rows[0][1] == 0
    assert rows[0][3] > rows[1][3] and rows[0][3] > rows[2][3]


def test_propose_excludes_neurons_already_in_the_set():
    pred = {"lean": [[-1.0, -0.5]], "acq": [[0.0, 0.0]], "dec": [[0.0, 0.0]]}
    rows = combo.propose(pred, 5, exclude={(0, 0)})
    assert [r[1] for r in rows] == [1]


def test_omp_recovers_a_sparse_combination():
    rng = np.random.default_rng(0)
    cols = rng.normal(size=(64, 200))
    cols /= np.linalg.norm(cols, axis=0)
    target = 0.8 * cols[:, 17] - 0.5 * cols[:, 42]
    fit, cosines = combo.omp(cols, target, 2)
    assert {c for c, _ in fit} == {17, 42}
    assert cosines[-1] == pytest.approx(1.0, abs=1e-6)
    assert cosines[0] < cosines[1]


def test_subset_keeps_every_other_topic():
    items = _items(4)
    sub = combo.subset(items, 0.5)
    assert {i["topic"] for i in sub} == {"t0", "t2"}
    assert combo.subset(items, 1.0) == items
