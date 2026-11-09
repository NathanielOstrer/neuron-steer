"""mix.py: spec parsing (no model) and stacking steers on the real model (--model)."""
import json
from pathlib import Path
import pytest

import mix


def test_parse_splits_neurons_from_features():
    n, f = mix.parse("15:4041:0.6, f9:19463:0.6,24:6441:-0.6")
    assert n == "15:4041:0.6,24:6441:-0.6"
    assert f == [(9, 19463, 0.6)]


def test_parse_empty_spec_is_baseline():
    assert mix.parse("") == ("", [])


def test_parse_keeps_negative_feature_doses():
    assert mix.parse("f16:7690:-0.5")[1] == [(16, 7690, -0.5)]


def test_parse_rejects_malformed_items():
    with pytest.raises(ValueError):
        mix.parse("15:4041")


def test_catalog_specs_all_parse():
    cat = json.loads((Path(__file__).parent.parent / "personas.json").read_text())
    for key in ("neurons", "features", "recipes"):
        for e in cat.get(key, []):
            n, f = mix.parse(e["spec"])
            assert n or f, e["name"]
            for item in filter(None, n.split(",")):
                L, idx, _ = item.split(":")
                assert 0 <= int(L) < 32 and 0 <= int(idx) < 9216, e["name"]


@pytest.mark.model
class TestOnModel:
    @pytest.fixture(scope="class")
    def loaded(self):
        mlx = pytest.importorskip("mlx.core")
        import mlx_steer
        tok, model = mlx_steer.load()
        tm = mlx_steer.text_model(model)
        return tok, model, tm, mlx_steer.Steering._medians(tok, model, tm)

    def first_logits(self, tok, model):
        import mlx.core as mx
        import mlx_steer
        ids = tok.encode(mlx_steer.prompt_for(tok, [{"role": "user", "content": "Who are you?"}]))
        return model(mx.array([ids]))[0, -1]

    def test_two_neurons_in_one_layer_stack_and_remove(self, loaded):
        """Regression: a second neuron in a steered layer used to read .weight off the wrapper."""
        import mlx.core as mx
        tok, model, tm, meds = loaded
        plain = self.first_logits(tok, model)
        one = mix.Mix(tok, model, tm, meds, "5:8879:0.6")
        a = self.first_logits(tok, model)
        one.remove()
        both = mix.Mix(tok, model, tm, meds, "5:8879:0.6,5:6573:0.7")
        b = self.first_logits(tok, model)
        both.remove()
        assert not mx.allclose(a, b)
        assert mx.allclose(self.first_logits(tok, model), plain)

    def test_scale_zero_is_the_plain_model(self, loaded):
        import mlx.core as mx
        tok, model, tm, meds = loaded
        plain = self.first_logits(tok, model)
        rows = mx.load(str(mix.ROWS))
        m = mix.Mix(tok, model, tm, meds, "6:7448:-0.8,f9:7620:0.5", rows=rows)
        m.scale = 0
        assert mx.allclose(self.first_logits(tok, model), plain)
        m.scale = 1
        assert not mx.allclose(self.first_logits(tok, model), plain)
        m.remove()

    def test_feature_and_neuron_mix_removes_cleanly(self, loaded):
        import mlx.core as mx
        tok, model, tm, meds = loaded
        plain = self.first_logits(tok, model)
        m = mix.Mix(tok, model, tm, meds, "15:4041:0.6,f9:19463:0.6")
        assert not mx.allclose(self.first_logits(tok, model), plain)
        m.remove()
        assert mx.allclose(self.first_logits(tok, model), plain)


def test_chat_presets_keep_what_held_up_and_drop_what_broke():
    cat = {"chat": [{"name": "orig", "spec": "14:8949:1", "label": "Original", "group": "Originals"}],
           "neurons": [{"name": "good", "spec": "1:2:0.8", "verdict": "clear"},
                       {"name": "dead", "spec": "1:3:0.8", "verdict": "dead"},
                       {"name": "orig", "spec": "9:9:1", "verdict": "clear"},
                       {"name": "ours", "spec": "f9:7620:0.5"}],
           "recipes": [{"name": "fused-pair", "spec": "1:2:0.6,3:4:0.6", "verdict": "fused", "parts": "a + b"},
                       {"name": "broken-pair", "spec": "1:2:1,3:4:1", "verdict": "broke"}]}
    ps = mix.chat_presets(cat)
    assert [p["id"] for p in ps] == ["orig", "good", "ours", "fused-pair"]
    assert ps[0]["spec"] == "14:8949:1"  # the chat entry wins over a same-named catalog entry
    assert ps[1]["group"] == "Single neurons" and ps[3]["group"] == "Mixes"
    assert ps[3]["note"] == "a + b" and ps[1]["label"] == "Good"


def test_real_catalog_has_golden_gate_and_no_broken_presets():
    ps = {p["id"]: p for p in mix.chat_presets()}
    assert ps["golden-gate"]["spec"] == "6:7448:-0.8,5:6203:-0.75,25:8324:-0.4"
    assert "evil-recipe" in ps and "naruto" not in ps and "backhanded-full" not in ps


def test_chat_rows_file_covers_every_feature_preset():
    mx = pytest.importorskip("mlx.core")
    rows = mx.load(str(mix.ROWS))
    for p in mix.chat_presets():
        for L, f, _ in mix.parse(p["spec"])[1]:
            assert f"{L}:{f}" in rows, p["id"]
            assert rows[f"{L}:{f}"].shape == (2560,)
