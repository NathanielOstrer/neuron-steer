"""sae_steer.py with random toy SAEs. The encoder test runs anywhere MLX does; steering tests need --model."""
import pytest

mx = pytest.importorskip("mlx.core")
import sae_steer


def toy_sae(tmp_path, d, m=64, k=4, layer=9, seed=0):
    mx.random.seed(seed)
    W = mx.random.normal((m, d))
    W = W / mx.linalg.norm(W, axis=1, keepdims=True)
    t = {"W_enc": W.T, "b_enc": mx.zeros((m,)), "W_dec": W, "b_dec": mx.zeros((d,)), "scale": mx.array([0.5])}
    path = tmp_path / f"sae_L{layer}.safetensors"
    mx.save_safetensors(str(path), t, metadata={"k": str(k), "layer": str(layer), "width": str(m)})
    return path


def test_encode_keeps_top_k_and_applies_scale(tmp_path):
    sae = sae_steer.SAE(toy_sae(tmp_path, d=16))
    assert (sae.k, sae.layer, sae.scale) == (4, 9, 0.5)
    x = mx.random.normal((5, 16))
    f = sae.encode(x)
    assert ((f > 0).sum(1) <= 4).all().item()
    # With zero biases the encoder is linear and TopK picks the same latents, so 2x in gives 2x out.
    assert mx.allclose(sae.encode(2 * x), 2 * f, atol=1e-4).item()


def test_load_saes_keys_by_layer(tmp_path):
    toy_sae(tmp_path, d=16, layer=9)
    toy_sae(tmp_path, d=16, layer=16)
    assert sorted(sae_steer.load_saes(tmp_path)) == [9, 16]


@pytest.mark.model
def test_feature_steer_zero_dose_and_remove_match_plain(tmp_path):
    import mlx_steer, politics
    from eval_politics import Scorer
    tok, model = mlx_steer.load()
    tm = mlx_steer.text_model(model)
    meds = mlx_steer.Steering._medians(tok, model, tm)
    sae = sae_steer.SAE(toy_sae(tmp_path, d=2560))
    texts = [i["text"] for i in politics.load_items()[:4:2]]
    score = lambda: [Scorer(tok, model).item(t) for t in texts]
    plain = score()
    st = sae_steer.FeatureSteering(model, tm, sae, 3, meds)
    st.scale = 0
    assert score() == plain
    st.scale = 1.0
    assert score() != plain
    st.remove()
    assert score() == plain
