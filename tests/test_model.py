"""Integration tests on the real 8-bit model. They load about 4.5 GB, so they only run with --model."""
import pytest

mx = pytest.importorskip("mlx.core")
import mlx_steer, politics
from eval_politics import Scorer

pytestmark = pytest.mark.model
TEXTS = [i["text"] for i in politics.load_items()[:6:2]]


@pytest.fixture(scope="module")
def loaded():
    return mlx_steer.load()


def scores(tok, model):
    s = Scorer(tok, model)
    return [s.item(t) for t in TEXTS]


def test_scores_are_deterministic_and_answer_cleanly(loaded):
    a, b = scores(*loaded), scores(*loaded)
    assert a == b
    assert all(cov > 0.9 for _, cov in a)


def test_dose_zero_and_remove_match_the_plain_model(loaded):
    tok, model = loaded
    plain = scores(tok, model)
    st = mlx_steer.Steering(tok, model, "14:8949:1", log=False)
    st.scale = 0
    assert scores(tok, model) == plain
    st.scale = 1
    assert scores(tok, model) != plain
    st.remove()
    assert scores(tok, model) == plain


def test_evil_steer_still_moves_the_eval(loaded):
    """The known evil neurons drive agreement up at dose 0.5 (seen on the full set)."""
    tok, model = loaded
    plain = scores(tok, model)
    st = mlx_steer.Steering(tok, model, mlx_steer.TWEET, log=False)
    st.scale = 0.5
    steered = scores(tok, model)
    st.remove()
    assert sum(p for p, _ in steered) > sum(p for p, _ in plain)
