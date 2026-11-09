"""SAE trainer pieces on CPU with toy sizes. Needs torch: uv run --with torch --with safetensors pytest."""
import argparse
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from sae.train import TopKSAE, Activations, Stop, save, load_sae


def test_encode_keeps_at_most_k_nonnegative_latents():
    torch.manual_seed(0)
    sae = TopKSAE(d=16, m=64, k=5)
    z, pre = sae.encode(torch.randn(10, 16))
    assert ((z > 0).sum(1) <= 5).all()
    assert (z >= 0).all()
    assert z.shape == pre.shape == (10, 64)


def test_decoder_rows_stay_unit_norm():
    sae = TopKSAE(d=16, m=64, k=5)
    with torch.no_grad():
        sae.W_dec.mul_(3)
    sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5)


def test_recovers_planted_sparse_features():
    """Data made of 3 of 48 hidden unit directions per sample. A trained SAE reconstructs it."""
    g = torch.Generator().manual_seed(0)
    d, n_true = 32, 48
    true = torch.randn(n_true, d, generator=g)
    true /= true.norm(dim=1, keepdim=True)

    def batch(n=512):
        idx = torch.stack([torch.randperm(n_true, generator=g)[:3] for _ in range(n)])
        coef = torch.rand(n, 3, generator=g) + 0.5
        return (coef[..., None] * true[idx]).sum(1)

    torch.manual_seed(0)
    sae = TopKSAE(d, 96, k=4)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    for _ in range(1500):
        loss, aux, fvu = sae(batch(), dead_after=50_000)
        opt.zero_grad()
        (loss + aux / 32).backward()
        opt.step()
        sae.normalize_decoder()
    _, _, fvu = sae(batch(2048))
    assert fvu < 0.1


def test_auxk_is_zero_with_no_dead_latents_and_positive_with_dead_ones():
    torch.manual_seed(0)
    sae = TopKSAE(d=16, m=64, k=4, k_aux=8)
    x = torch.randn(32, 16)
    _, aux, _ = sae(x, dead_after=10**9)
    assert aux.item() == 0
    sae.since_fired.fill_(10**6)
    sae.since_fired[:8] = 0
    _, aux, _ = sae(x, dead_after=10)
    assert aux.item() > 0


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    sae = TopKSAE(d=8, m=24, k=3)
    args = argparse.Namespace(width=24)
    save({5: sae}, {5: 0.25}, args, tmp_path)
    loaded, scale = load_sae(tmp_path / "sae_L5.safetensors")
    assert scale == 0.25
    assert loaded.k == 3
    for name in ("W_enc", "W_dec", "b_enc", "b_dec"):
        assert torch.equal(getattr(loaded, name), getattr(sae, name))


class ToyBlock(torch.nn.Module):
    def __init__(self, add):
        super().__init__()
        self.add, self.calls = add, 0

    def forward(self, h):
        self.calls += 1
        return (h + self.add,)  # tuple output, like HF decoder layers


class ToyModel(torch.nn.Module):
    def __init__(self, n=4, d=3):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([ToyBlock(i + 1) for i in range(n)])
        self.d = d

    def forward(self, input_ids, use_cache=False):
        h = input_ids[..., None].float().expand(*input_ids.shape, self.d)
        for b in self.model.layers:
            h = b(h)[0]
        return h


def test_activations_drop_position_zero_and_stop_after_deepest_layer():
    m = ToyModel()
    acts = Activations(m, [0, 2])
    ids = torch.tensor([[10, 20, 30], [40, 50, 60]])
    out = acts(ids)
    # layer 0 adds 1, layer 2 output adds 1+2+3 = 6. Position 0 (10 and 40) is gone.
    assert out[0][:, 0].tolist() == [21, 31, 51, 61]
    assert out[2][:, 0].tolist() == [26, 36, 56, 66]
    assert m.model.layers[3].calls == 0
