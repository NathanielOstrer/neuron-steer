"""Find and steer SAE features on MLX, using SAEs trained by sae/train.py.

    .venv/bin/python sae_steer.py rank --sae-dir sae_out                 # contrast + attribution per layer
    .venv/bin/python sae_steer.py show --sae-dir sae_out --feature 16:1234   # top-activating contexts
    .venv/bin/python sae_steer.py screen --sae-dir sae_out --only 16:1234,9:77 --doses=-0.3,0.3

Feature steering adds dose * median(||resid leaving L||) * u to the residual leaving layer L at
every position, where u is the feature's unit decoder row. That is the neuron recipe with
W_down[:, n] / ||W_down[:, n]|| replaced by u, so doses are comparable between the two.
Ranking uses train items only, as in find_neurons.py.
"""
import argparse, json, sys, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx_steer, politics
from mlx_steer import ResidAdd, wrap  # moved there so serve.py can steer features (kept importable here)
from eval_politics import Scorer, OUT, fmt


class SAE:
    def __init__(self, path):
        t = mx.load(str(path))
        self.W_enc, self.b_enc, self.W_dec, self.b_dec = t["W_enc"], t["b_enc"], t["W_dec"], t["b_dec"]
        self.scale = t["scale"].item()
        meta = mx.load(str(path), return_metadata=True)[1]
        self.k, self.layer = int(meta["k"]), int(meta["layer"])

    def encode(self, x):
        """x: (n, d) raw residuals. Returns dense (n, m) feature activations."""
        pre = (x.astype(mx.float32) * self.scale - self.b_dec) @ self.W_enc + self.b_enc
        kth = mx.sort(pre, axis=-1)[:, -self.k][:, None]
        return mx.where(pre >= kth, mx.maximum(pre, 0), 0)


def load_saes(d):
    return {s.layer: s for s in (SAE(p) for p in sorted(Path(d).glob("sae_L*.safetensors")))}


def rank(args):
    tok, model = mlx_steer.load()
    tm = mlx_steer.text_model(model)
    meds = mlx_steer.Steering._medians(tok, model, tm)
    saes = load_saes(args.sae_dir)
    taps, orig = wrap(tm, saes)
    items = politics.load_items(split="train")
    out = {"meds": meds}

    # Contrast: feature activations at the last prompt token under the two personas.
    means = {}
    for side, persona in politics.PERSONAS.items():
        scorer = Scorer(tok, model, persona)
        acc = {L: [] for L in saes}
        for it in items:
            for ids in scorer.ids(it["text"]):
                model(mx.array([ids]))
                for L in saes:
                    acc[L].append(taps[L].last)
        means[side] = {}
        for L, sae in saes.items():
            f = sae.encode(mx.stack(acc[L]))
            means[side][L] = (f.mean(0), (f * f).mean(0) - f.mean(0) ** 2, (f > 0).mean(0))
            mx.eval(means[side][L])
        print(f"{side} persona done", flush=True)

    # Attribution: d(lean)/d(dose) for adding med_L * u_f at every position.
    # The gradient w.r.t. one vector g added to the residual gives all features at once: g . u_f.
    scorer = Scorer(tok, model)
    A, D = mx.array(scorer.agree), mx.array(scorer.disagree)
    model.train()  # plain-ops linear attention, which has a backward pass

    def objective(vecs, ids, sign):
        for L, v in zip(saes, vecs):
            taps[L].vec, taps[L].scale = v, 1.0
        logits = model(ids)[0, -1].astype(mx.float32)
        return sign * (mx.logsumexp(logits[A]) - mx.logsumexp(logits[D]))

    grad_fn = mx.grad(objective)
    d = saes[next(iter(saes))].W_dec.shape[1]
    zeros = [mx.zeros((d,), dtype=mx.float32) for _ in saes]
    total, n, t = None, 0, time.time()
    mx.set_cache_limit(512 * 2**20)
    for it in items:
        sign = 1.0 if it["side"] == "left" else -1.0
        for ids in scorer.ids(it["text"]):
            g = mx.stack(grad_fn(zeros, mx.array([ids]), sign))
            total = g if total is None else total + g
            mx.eval(total)
            mx.clear_cache()
            n += 1
        if n % 60 == 0:
            print(f"attribution {n} prompts, {time.time() - t:.0f}s", flush=True)
    model.eval()
    for L in saes:
        taps[L].vec = None
    tm.layers = orig

    for i, (L, sae) in enumerate(saes.items()):
        mL, vL, fL = means["left"][L]
        mR, vR, fR = means["right"][L]
        contrast = (mL - mR) / mx.sqrt((vL + vR) / 2 + 1e-6)
        effect = 2 * (sae.W_dec @ (total[i] / n)) * meds[L]  # predicted lean_lo change at dose +1
        out[f"L{L}"] = {
            "contrast": top(contrast, args.k, extra={"fires_left": fL, "fires_right": fR}),
            "attribution": top(effect, args.k),
        }
        show(f"L{L} contrast (standardized diff, left persona minus right)", out[f"L{L}"]["contrast"])
        show(f"L{L} attribution (predicted lean_lo change at dose +1)", out[f"L{L}"]["attribution"])
    (OUT / "sae-candidates.json").write_text(json.dumps(out, indent=1))


def top(score, k, extra=None):
    idx = mx.argsort(-mx.abs(score))[:k].tolist()
    rows = []
    for i in idx:
        r = {"feature": i, "score": score[i].item()}
        for name, arr in (extra or {}).items():
            r[name] = round(arr[i].item(), 3)
        rows.append(r)
    return rows


def show(title, rows, n=12):
    print(f"\n{title}")
    for r in rows[:n]:
        rest = " ".join(f"{k} {v}" for k, v in r.items() if k not in ("feature", "score"))
        print(f"  F{r['feature']:<6} {r['score']:+.3f}  {rest}")


def contexts(tok, sae_dir, L, f, n=8, width=12):
    """Top-activating contexts for feature f from top_L{L}.safetensors and tokens.npy."""
    import numpy as np
    t = mx.load(str(Path(sae_dir) / f"top_L{L}.safetensors"))
    corpus = np.load(Path(sae_dir) / "tokens.npy").reshape(-1)
    vals, idx = t["values"][f].tolist(), t["token_index"][f].tolist()
    rows = []
    for v, i in list(zip(vals, idx))[:n]:
        if v <= 0:
            break
        before = tok.decode(corpus[max(0, i - width):i].tolist())
        rows.append({"act": round(v, 2), "before": before, "token": tok.decode([int(corpus[i])]),
                     "after": tok.decode(corpus[i + 1:i + 4].tolist())})
    return rows, t["density"][f].item()


def show_feature(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mlx_steer.MID)
    for spec in args.feature.split(","):
        L, f = map(int, spec.split(":"))
        rows, density = contexts(tok, args.sae_dir, L, f, n=args.n)
        print(f"\nL{L} F{f}  fires on {density:.4%} of tokens")
        for r in rows:
            ctx = (r["before"] + "[[" + r["token"] + "]]" + r["after"]).replace("\n", "\\n")
            print(f"  {r['act']:6.2f}  {ctx}")


class FeatureSteering:
    """Steer one SAE feature: `scale` multiplies the dose; 0 turns it off. remove() restores the model."""

    def __init__(self, model, tm, sae, f, meds):
        self.tm, self.orig = tm, list(tm.layers)
        self.taps, _ = wrap(tm, [sae.layer])
        self.taps[sae.layer].vec = sae.W_dec[f] * meds[sae.layer]

    @property
    def scale(self):
        return next(iter(self.taps.values())).scale

    @scale.setter
    def scale(self, v):
        for t in self.taps.values():
            t.scale = v

    def remove(self):
        self.tm.layers = self.orig


def screen(args):
    tok, model = mlx_steer.load()
    tm = mlx_steer.text_model(model)
    meds = mlx_steer.Steering._medians(tok, model, tm)
    saes = load_saes(args.sae_dir)
    items = politics.load_items(split=args.split)
    if args.subset:
        topics = sorted({i["topic"] for i in items})[::2]
        items = [i for i in items if i["topic"] in topics]
    scorer = Scorer(tok, model)
    base, _ = scorer.run(items)
    print(f"base: {fmt(base)}", flush=True)
    results = []
    for spec in args.only.split(","):
        L, f = map(int, spec.split(":"))
        st = FeatureSteering(model, tm, saes[L], f, meds)
        row, line = {"layer": L, "feature": f}, f"L{L:<2} F{f:<6}"
        for dose in [float(x) for x in args.doses.split(",")]:
            st.scale = dose
            s, _ = scorer.run(items)
            row[f"{dose:+g}"] = {k: s[k] for k in ("lean", "lean_lo", "lean_econ", "lean_social", "acquiescence",
                                                    "decisiveness", "coverage", "topics")}
            line += f" {dose:+g}: {s['lean']:+.2f}/{s['decisiveness']:.2f}/{s['coverage']:.2f}"
        st.remove()
        results.append(row)
        print(line, flush=True)
    (OUT / f"sae-screen-{args.name}.json").write_text(json.dumps({"base": base, "results": results}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rank", "show", "screen"])
    ap.add_argument("--sae-dir", default="sae_out")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--feature", help="L:F,... for show")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--only", help="L:F,... for screen")
    ap.add_argument("--doses", default="-0.3,-0.15,0.15,0.3")
    ap.add_argument("--split", default="train", choices=["all", "train", "test"])
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--name", default="top")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"rank": rank, "show": show_feature, "screen": screen}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
