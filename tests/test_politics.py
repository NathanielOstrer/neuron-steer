import math
import pytest
import politics


def item(topic, side, i=0, axis="econ", split="train"):
    return {"id": f"{topic}.{side[0]}{i}", "topic": topic, "axis": axis, "split": split, "side": side, "text": "x"}


def test_shipped_items_are_balanced():
    items = politics.load_items()
    assert politics.check_items(items) == []
    assert len(items) == 240
    assert {i["split"] for i in items} == {"train", "test"}


def test_split_filter():
    train, test = politics.load_items(split="train"), politics.load_items(split="test")
    assert len(train) + len(test) == 240
    assert not {i["topic"] for i in train} & {i["topic"] for i in test}


@pytest.mark.parametrize("bad, msg", [
    ([item("a", "left"), item("a", "left", 1)], "2 left vs 0 right"),
    ([item("a", "left"), item("a", "right", split="test")], "more than one split"),
    ([item("a", "left"), item("a", "right", axis="social")], "more than one axis"),
    ([item("a", "left"), item("a", "left")], "duplicate id"),
    ([item("a", "centre")], "bad side"),
])
def test_check_items_catches_problems(bad, msg):
    assert any(msg in p for p in politics.check_items(bad))


def test_answer_probs_normalizes_over_the_two_answers():
    probs = [0.0] * 10
    probs[1], probs[2], probs[3] = 0.3, 0.1, 0.4  # agree ids 1, 2; disagree id 3
    p, cov = politics.answer_probs(probs, [1, 2], [3])
    assert p == pytest.approx(0.5)
    assert cov == pytest.approx(0.8)


def test_agree_with_everything_scores_zero_lean():
    items = [item("a", "left"), item("a", "right"), item("b", "left", axis="social"), item("b", "right", axis="social")]
    s = politics.summarize(items, {i["id"]: 0.9 for i in items})
    assert s["lean"] == pytest.approx(0)
    assert s["lean_lo"] == pytest.approx(0)
    assert s["acquiescence"] == pytest.approx(0.4)


def test_lean_sign_and_axes():
    items = [item("a", "left"), item("a", "right"), item("b", "left", axis="social"), item("b", "right", axis="social")]
    scores = {"a.l0": 0.8, "a.r0": 0.2, "b.l0": 0.3, "b.r0": 0.7}
    s = politics.summarize(items, scores)
    assert s["lean_econ"] == pytest.approx(0.6)  # agrees with left item: positive
    assert s["lean_social"] == pytest.approx(-0.4)
    assert s["lean"] == pytest.approx(0.1)
    assert s["lean_lo_econ"] == pytest.approx(2 * math.log(4))
    assert s["topics"] == {"a": pytest.approx(0.6), "b": pytest.approx(-0.4)}


def test_logodds_clamps_extremes():
    assert math.isfinite(politics.logodds(0.0))
    assert math.isfinite(politics.logodds(1.0))
    assert politics.logodds(0.5) == 0


def test_decisiveness():
    items = [item("a", "left"), item("a", "right")]
    assert politics.summarize(items, {"a.l0": 1.0, "a.r0": 0.0})["decisiveness"] == pytest.approx(1)
    assert politics.summarize(items, {"a.l0": 0.5, "a.r0": 0.5})["decisiveness"] == pytest.approx(0)
    assert politics.summarize(items, {"a.l0": 0.75, "a.r0": 0.25})["decisiveness"] == pytest.approx(0.5)
