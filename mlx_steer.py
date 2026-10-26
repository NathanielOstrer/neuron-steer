"""The steer.py recipe on MLX, for the chat server. 8-bit weights use about 4.5 GB instead of 9 GB.

Same recipe: add dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
to neuron n at the down_proj input, at every position.
"""
import sys
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load

MID = "mlx-community/Qwen3.5-4B-8bit"
TWEET = "14:8949:1,13:7956:-1"
CALIB = ("The history of the printing press begins in the fifteenth century. "
         "Johannes Gutenberg combined movable metal type with oil-based ink and a "
         "screw press adapted from wine making, and books became cheap enough to spread widely.")


def text_model(model):
    """The decoder stack that owns the layer list (not a wrapper exposing it as a property)."""
    m = model
    while True:
        nxt = next((getattr(m, a) for a in ("language_model", "model") if hasattr(getattr(m, a, None), "layers")), None)
        if nxt is None:
            return m
        m = nxt


def down_weight(proj):
    """Full-precision W_down, dequantized if needed. Shape (hidden, intermediate)."""
    if hasattr(proj, "scales"):
        return mx.dequantize(proj.weight, proj.scales, getattr(proj, "biases", None),
                             proj.group_size, proj.bits)
    return proj.weight


class SteeredDown(nn.Module):
    def __init__(self, inner, onehot_delta, steering):
        super().__init__()
        self.inner, self.vec, self.steering = inner, onehot_delta, steering

    def __call__(self, x):
        if self.steering.scale:
            x = x + (self.vec * self.steering.scale).astype(x.dtype)
        return self.inner(x)


class Steering:
    """Wraps down_proj on the steered layers. `scale` multiplies every dose; 0 turns steering off."""

    def __init__(self, tok, model, spec=TWEET, log=True):
        self.scale = 1.0
        tm = text_model(model)
        meds = self._medians(tok, model, tm)
        for item in spec.split(","):
            L, n, dose = item.split(":")
            L, n, dose = int(L), int(n), float(dose)
            mlp = tm.layers[L].mlp
            W = down_weight(mlp.down_proj).astype(mx.float32)
            wnorm = mx.linalg.norm(W[:, n]).item()
            delta = dose * meds[L] / wnorm
            vec = mx.zeros((W.shape[1],), dtype=mx.float32)
            vec[n] = delta
            mlp.down_proj = SteeredDown(mlp.down_proj, vec, self)
            if log:
                print(f"L{L} N{n} dose {dose:+}: median resid {meds[L]:.1f}, "
                      f"|W_down[:,n]| {wnorm:.4f}, add {delta:+.1f}", file=sys.stderr)

    @staticmethod
    def _medians(tok, model, tm):
        """Median residual norm leaving each layer, tokens 1:, from one plain forward pass."""
        outs, orig = [], list(tm.layers)

        class Record(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.is_linear = getattr(inner, "is_linear", False)

            def __call__(self, *a, **k):
                h = self.inner(*a, **k)
                outs.append(h)
                return h

        tm.layers = [Record(l) for l in orig]
        try:
            model(mx.array([tok.encode(CALIB)]))
        finally:
            tm.layers = orig
        meds = []
        for h in outs:
            norms = mx.sort(mx.linalg.norm(h[0, 1:].astype(mx.float32), axis=-1))
            n = norms.shape[0]
            meds.append(float(((norms[(n - 1) // 2] + norms[n // 2]) / 2).item()))
        return meds


def load():
    model, tok = mlx_load(MID)
    return tok, model


def prompt_for(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
