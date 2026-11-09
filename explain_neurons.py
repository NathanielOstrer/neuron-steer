"""What does an MLP neuron do? Three views per neuron, on MLX.

    .venv/bin/python explain_neurons.py                                   # the evil pair + political candidates
    .venv/bin/python explain_neurons.py --neurons 14:8949,9:7944 --seqs 500

1. Top contexts: the down_proj input (the neuron's activation) over the first --seqs
   sequences of sae_out/tokens.npy, most positive and most negative.
2. Logit lens: W_down[:, n] through the final norm and the unembedding, top and bottom tokens.
3. SAE match: cosine of W_down[:, n] with the decoder rows of the first trained SAE at or after
   layer L, since the write persists in the residual stream until that SAE reads it.
Writes runs/neurons/<name>.json.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx_steer, sae_steer

DEFAULT = "14:8949,13:7956,9:7944,17:3980,17:3779,15:1515"
OUT = Path("runs/neurons")


class Stop(Exception):
    pass


class Record(nn.Module):
    """Wraps down_proj and keeps the input at the chosen neurons, shape (B, T, len(idx))."""

    def __init__(self, inner, idx):
        super().__init__()
        self.inner, self.idx, self.acts = inner, mx.array(idx), None

    def __call__(self, x):
        self.acts = x[..., self.idx].astype(mx.float32)
        return self.inner(x)


class StopAt(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.is_linear = getattr(inner, "is_linear", False)

    def __call__(self, *a, **k):
        raise Stop


def scan(model, tm, neurons, corpus, batch, n_top):
    """Top and bottom activations per neuron. Returns {(L, n): (vals_hi, idx_hi, vals_lo, idx_lo, mean, frac_pos)}."""
    by_layer = {}
    for L, n in neurons:
        by_layer.setdefault(L, []).append(n)
    deepest = max(by_layer)
    orig_layers = list(tm.layers)
    recs = {}
    for L, ns in by_layer.items():
        mlp = tm.layers[L].mlp
        recs[L] = (mlp, mlp.down_proj)
        mlp.down_proj = Record(mlp.down_proj, ns)
    if deepest + 1 < len(orig_layers):
        layers = list(orig_layers)
        layers[deepest + 1] = StopAt(orig_layers[deepest + 1])
        tm.layers = layers
    S, T = corpus.shape
    hi = {k: (np.full(n_top, -np.inf), np.zeros(n_top, np.int64)) for k in neurons}
    lo = {k: (np.full(n_top, np.inf), np.zeros(n_top, np.int64)) for k in neurons}
    total = {k: 0.0 for k in neurons}
    pos = {k: 0 for k in neurons}
    count = 0
    t0 = time.time()
    try:
        for s in range(0, S, batch):
            ids = mx.array(corpus[s:s + batch])
            try:
                model(ids)
            except Stop:
                pass
            for L, ns in by_layer.items():
                a = np.array(recs[L][0].down_proj.acts)  # (B, T, k)
                a[:, 0] = np.nan  # position 0 is an outlier sink, as in the SAE data
                flat = a.reshape(-1, len(ns))
                base = s * T
                for j, n in enumerate(ns):
                    v = flat[:, j]
                    ok = ~np.isnan(v)
                    total[(L, n)] += float(v[ok].sum())
                    pos[(L, n)] += int((v[ok] > 0).sum())
                    vv = np.where(ok, v, -np.inf)
                    top = np.argpartition(-vv, n_top)[:n_top]
                    hv, hi_idx = hi[(L, n)]
                    cv, ci = np.concatenate([hv, vv[top]]), np.concatenate([hi_idx, top + base])
                    keep = np.argsort(-cv)[:n_top]
                    hi[(L, n)] = (cv[keep], ci[keep])
                    vv = np.where(ok, v, np.inf)
                    bot = np.argpartition(vv, n_top)[:n_top]
                    lv, li = lo[(L, n)]
                    cv, ci = np.concatenate([lv, vv[bot]]), np.concatenate([li, bot + base])
                    keep = np.argsort(cv)[:n_top]
                    lo[(L, n)] = (cv[keep], ci[keep])
            count += ids.shape[0] * (T - 1)
            mx.clear_cache()
            if (s // batch) % 25 == 0:
                print(f"scanned {count:,} tokens, {count / (time.time() - t0):,.0f} tok/s", file=sys.stderr, flush=True)
    finally:
        tm.layers = orig_layers
        for mlp, inner in recs.values():
            mlp.down_proj = inner
    return {k: (*hi[k], *lo[k], total[k] / count, pos[k] / count) for k in neurons}


def context(tok, flat, i, before=14, after=4):
    return {"before": tok.decode(flat[max(0, i - before):i].tolist()), "token": tok.decode([int(flat[i])]),
            "after": tok.decode(flat[i + 1:i + 1 + after].tolist())}


def logit_lens(model, tm, tok, w, k):
    lm = next(m for m in (getattr(model, "language_model", None), model) if m is not None and hasattr(m, "args"))
    h = tm.norm(w[None].astype(mx.float32))
    logits = (lm.lm_head(h) if not lm.args.tie_word_embeddings else tm.embed_tokens.as_linear(h))[0]
    order = mx.argsort(logits).tolist()
    show = lambda ids: [tok.decode([i]) for i in ids]
    return {"up": show(order[::-1][:k]), "down": show(order[:k])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", default=DEFAULT, help="L:N,...")
    ap.add_argument("--seqs", type=int, default=2000, help="512-token sequences of the corpus to scan")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-top", type=int, default=12)
    ap.add_argument("--sae-dir", default="sae_out")
    ap.add_argument("--name", default="explain")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(512 * 2**20)
    neurons = [tuple(map(int, s.split(":"))) for s in args.neurons.split(",")]
    tok, model = mlx_steer.load()
    tm = mlx_steer.text_model(model)
    corpus = np.load(Path(args.sae_dir) / "tokens.npy")[:args.seqs]
    flat = corpus.reshape(-1)
    stats = scan(model, tm, neurons, corpus, args.batch, args.n_top)
    saes = sae_steer.load_saes(args.sae_dir)
    out = []
    for L, n in neurons:
        vh, ih, vl, il, mean, frac = stats[(L, n)]
        w = mlx_steer.down_weight(tm.layers[L].mlp.down_proj).astype(mx.float32)[:, n]
        w = w / mx.linalg.norm(w)
        row = {"layer": L, "neuron": n, "mean": mean, "frac_pos": frac,
               "top": [{"act": round(float(v), 3)} | context(tok, flat, int(i)) for v, i in zip(vh, ih)],
               "bottom": [{"act": round(float(v), 3)} | context(tok, flat, int(i)) for v, i in zip(vl, il)],
               "logit_lens": logit_lens(model, tm, tok, w, 15)}
        sl = next((l for l in sorted(saes) if l >= L), None)
        if sl is not None:
            cos = saes[sl].W_dec @ w
            idx = mx.argsort(-mx.abs(cos))[:5].tolist()
            row["sae"] = {"layer": sl, "features": []}
            for f in idx:
                ctx, density = sae_steer.contexts(tok, args.sae_dir, sl, f, n=3)
                row["sae"]["features"].append({"feature": f, "cos": round(cos[f].item(), 3), "density": density,
                                               "contexts": ctx})
        out.append(row)
        print(f"\n=== L{L} N{n}  mean {mean:+.3f}  positive on {frac:.1%} of tokens")
        for side in ("top", "bottom"):
            print(f"  {side}:")
            for r in row[side][:6]:
                print(f"    {r['act']:+7.2f}  " + (r["before"] + "[[" + r["token"] + "]]" + r["after"]).replace("\n", "\\n"))
        print(f"  logit lens up:   {row['logit_lens']['up']}")
        print(f"  logit lens down: {row['logit_lens']['down']}")
        for fr in row.get("sae", {}).get("features", []):
            c = fr["contexts"][0] if fr["contexts"] else None
            ex = (c["before"][-40:] + "[[" + c["token"] + "]]" + c["after"]).replace("\n", "\\n") if c else ""
            print(f"  SAE L{row['sae']['layer']} F{fr['feature']:<6} cos {fr['cos']:+.3f}  {ex}")
    (OUT / f"{args.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
