"""explain_neurons.py. The context test runs anywhere MLX does; the scan test needs --model."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import explain_neurons


class FakeTok:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def test_context_clips_at_corpus_start():
    flat = np.arange(10)
    c = explain_neurons.context(FakeTok(), flat, 2, before=5, after=3)
    assert c == {"before": "<0><1>", "token": "<2>", "after": "<3><4><5>"}


@pytest.mark.model
def test_scan_matches_direct_forward_and_restores_model():
    import mlx_steer
    tok, model = mlx_steer.load()
    tm = mlx_steer.text_model(model)
    corpus = np.array([tok.encode("The quick brown fox jumps over the lazy dog. " * 4)[:32]] * 2)
    corpus[1] = corpus[1][::-1]
    plain = model(mx.array(corpus[:1]))
    stats = explain_neurons.scan(model, tm, [(9, 7944), (14, 8949)], corpus, batch=2, n_top=3)
    # The model is restored: same logits, original layer and down_proj objects.
    assert mx.array_equal(model(mx.array(corpus[:1])), plain).item()
    assert not isinstance(tm.layers[15], explain_neurons.StopAt)
    assert not isinstance(tm.layers[9].mlp.down_proj, explain_neurons.Record)
    # The top value matches a direct read of the activation, and position 0 is never picked.
    vh, ih, vl, il, mean, frac = stats[(9, 7944)]
    assert all(vh[i] >= vh[i + 1] for i in range(2)) and all(vl[i] <= vl[i + 1] for i in range(2))
    assert all(i % 32 != 0 for i in list(ih) + list(il))
    rec = explain_neurons.Record(tm.layers[9].mlp.down_proj, [7944])
    tm.layers[9].mlp.down_proj = rec
    try:
        model(mx.array(corpus))
    finally:
        tm.layers[9].mlp.down_proj = rec.inner
    direct = np.array(rec.acts).reshape(-1)
    assert np.isclose(direct[ih[0]], vh[0], atol=1e-3)
    assert 0 <= frac <= 1
