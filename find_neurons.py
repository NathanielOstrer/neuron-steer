"""Rank neurons that might carry political lean, then check the top ones by steering.

    .venv/bin/python find_neurons.py contrast   # methods 1 and 2: Democrat vs Republican persona
    .venv/bin/python find_neurons.py attrib     # method 3: gradient of the lean w.r.t. each neuron
    .venv/bin/python find_neurons.py screen --top 10   # steer the top candidates on train items
    .venv/bin/python find_neurons.py screen --only 9:7944,15:1515 --doses 0.15,-0.15

All ranking uses the train split only. Results go to runs/politics/.

Method 1 (direction): diff of mean residual leaving layer L, left persona minus right,
at the last prompt token. Score = cos(W_down[:, n], direction). Positive means a
positive dose writes toward the Democrat persona.
Method 2 (activation): standardized diff of the neuron's down_proj input under the two personas.
Method 3 (attribution): d(lean)/d(neuron input, summed over positions) times the recipe's
delta. This is a first-order estimate of how much dose +1 moves lean_lo, with no persona.
"""
import argparse, json, sys, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx_steer, politics
from eval_politics import Scorer, OUT, fmt


class Tap(nn.Module):
    """Wraps down_proj: records the last-position input, and adds `pert` (broadcast over positions)."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.last, self.pert = inner, None, None

    def __call__(self, x):
        if self.pert is not None:
            x = x + self.pert.astype(x.dtype)
        self.last = x[0, -1]
        return self.inner(x)


class LayerTap(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.last = inner, None
        self.is_linear = getattr(inner, "is_linear", False)

    def __call__(self, *a, **k):
        h = self.inner(*a, **k)
        self.last = h[0, -1]
        return h


def setup():
    tok, model = mlx_steer.load()
    tm = mlx_steer.text_model(model)
    meds = mlx_steer.Steering._medians(tok, model, tm)
    wnorm = []
    for layer in tm.layers:
        W = mlx_steer.down_weight(layer.mlp.down_proj).astype(mx.float32)
        wnorm.append(mx.linalg.norm(W, axis=0))
        mx.eval(wnorm[-1])
    return tok, model, tm, meds, wnorm


def install_taps(tm):
    taps = []
    for layer in tm.layers:
        layer.mlp.down_proj = Tap(layer.mlp.down_proj)
        taps.append(layer.mlp.down_proj)
    orig = list(tm.layers)
    tm.layers = [LayerTap(l) for l in orig]
    return taps, list(tm.layers), orig


def contrast(args):
    tok, model, tm, meds, wnorm = setup()
    taps, ltaps, orig = install_taps(tm)
    items = politics.load_items(split="train")
    stats = {}
    for side, persona in politics.PERSONAS.items():
        scorer = Scorer(tok, model, persona)
        n, s_act, s_act2, s_res = 0, None, None, None
        t = time.time()
        for it in items:
            for ids in scorer.ids(it["text"]):
                model(mx.array([ids]))
                act = mx.stack([tp.last.astype(mx.float32) for tp in taps])
                res = mx.stack([lt.last.astype(mx.float32) for lt in ltaps])
                s_act = act if s_act is None else s_act + act
                s_act2 = act * act if s_act2 is None else s_act2 + act * act
                s_res = res if s_res is None else s_res + res
                mx.eval(s_act, s_act2, s_res)
                n += 1
        stats[side] = (s_act / n, s_act2 / n - (s_act / n) ** 2, s_res / n)
        print(f"{side}: {n} prompts in {time.time() - t:.0f}s", flush=True)
    tm.layers = orig
    (mL, vL, rL), (mR, vR, rR) = stats["left"], stats["right"]
    d_act = (mL - mR) / mx.sqrt((vL + vR) / 2 + 1e-6)
    direction = rL - rR  # (layers, hidden)
    cos = []
    for i, layer in enumerate(tm.layers):
        W = mlx_steer.down_weight(taps[i].inner).astype(mx.float32)  # (hidden, inter)
        d = direction[i] / mx.linalg.norm(direction[i])
        cos.append((d @ W) / wnorm[i])
        mx.eval(cos[-1])
    cos = mx.stack(cos)
    out = {"meds": meds, "dir_norm": mx.linalg.norm(direction, axis=1).tolist(),
           "direction": top(cos, args.k), "activation": top(d_act, args.k)}
    (OUT / "candidates-contrast.json").write_text(json.dumps(out, indent=1))
    print("residual diff norm by layer:", " ".join(f"{x:.1f}" for x in out["dir_norm"]))
    show("method 1 (direction cos)", out["direction"])
    show("method 2 (activation d')", out["activation"])


def attrib(args):
    # Without a small cache, freed backward buffers pile up and the mini starts swapping:
    # each pass then takes 10-50 s instead of 2 s.
    mx.set_cache_limit(512 * 2**20)
    tok, model, tm, meds, wnorm = setup()
    taps, _, orig = install_taps(tm)
    tm.layers = orig  # only the down_proj taps are needed
    # The fused linear-attention kernel has no backward pass. Training mode switches
    # Qwen3.5 to the plain-ops path, which does (the model has no dropout).
    model.train()
    scorer = Scorer(tok, model)
    A = mx.array(scorer.agree)
    D = mx.array(scorer.disagree)
    inter = wnorm[0].shape[0]

    def objective(perts, ids, sign):
        for tp, p in zip(taps, perts):
            tp.pert = p
        logits = model(ids)[0, -1].astype(mx.float32)
        lo = mx.logsumexp(logits[A]) - mx.logsumexp(logits[D])
        return sign * lo

    grad_fn = mx.grad(objective)
    zeros = [mx.zeros((inter,), dtype=mx.float32) for _ in taps]
    total, n, t = None, 0, time.time()
    items = politics.load_items(split="train")
    for it in items:
        sign = 1.0 if it["side"] == "left" else -1.0
        for ids in scorer.ids(it["text"]):
            g = mx.stack(grad_fn(zeros, mx.array([ids]), sign))
            total = g if total is None else total + g
            mx.eval(total)
            del g
            mx.clear_cache()
            n += 1
        if n % 40 == 0:
            print(f"{n} prompts, {time.time() - t:.0f}s", flush=True)
    for tp in taps:
        tp.pert = None
    # lean_lo = mean(left lo) - mean(right lo) = 2 * mean(sign * lo) on a balanced set.
    delta = mx.stack([meds[i] / wnorm[i] for i in range(len(taps))])
    effect = 2 * (total / n) * delta
    out = {"attribution": top(effect, args.k)}
    (OUT / "candidates-attrib.json").write_text(json.dumps(out, indent=1))
    show("method 3 (predicted lean_lo change at dose +1)", out["attribution"])


def top(score, k):
    """Top k by |score| as [layer, neuron, score]."""
    flat = score.reshape(-1)
    idx = mx.argsort(-mx.abs(flat))[:k].tolist()
    inter = score.shape[1]
    return [[i // inter, i % inter, float(flat[i].item())] for i in idx]


def show(title, rows, n=15):
    print(f"\n{title}")
    for L, nrn, s in rows[:n]:
        print(f"  L{L:<2} N{nrn:<5} {s:+.3f}")


def screen(args):
    """Steer each candidate at each dose on train items; prints lean/coverage per dose."""
    tok, model = mlx_steer.load()
    tm = mlx_steer.text_model(model)
    meds = mlx_steer.Steering._medians(tok, model, tm)
    items = politics.load_items(split="train")
    if args.subset:
        topics = sorted({i["topic"] for i in items})[::2]
        items = [i for i in items if i["topic"] in topics]
    scorer = Scorer(tok, model)
    cands = [(int(L), int(n), "manual", 0.0) for L, n in (c.split(":") for c in args.only.split(","))] if args.only else []
    for f in [] if args.only else args.files:
        data = json.loads((OUT / f).read_text())
        for method in ("direction", "activation", "attribution"):
            for L, n, s in data.get(method, [])[:args.top]:
                if (L, n) not in [(c[0], c[1]) for c in cands]:
                    cands.append((L, n, method, s))
    base, _ = scorer.run(items)
    print(f"base: {fmt(base)}", flush=True)
    results = []
    for L, n, method, s in cands:
        st = mlx_steer.Steering(tok, model, f"{L}:{n}:1", log=False, meds=meds)
        row = {"layer": L, "neuron": n, "method": method, "score": s}
        line = f"L{L:<2} N{n:<5} {method:<11} {s:+.3f} "
        for dose in [float(d) for d in args.doses.split(",")]:
            st.scale = dose
            summ, _ = scorer.run(items)
            row[f"{dose:+g}"] = {k: summ[k] for k in ("lean", "lean_lo", "acquiescence", "coverage")}
            line += f" {dose:+g}: {summ['lean']:+.2f}/{summ['coverage']:.2f}"
        st.remove()
        results.append(row)
        print(line, flush=True)
    (OUT / f"screen-{args.name}.json").write_text(json.dumps({"base": base, "results": results}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["contrast", "attrib", "screen"])
    ap.add_argument("--k", type=int, default=200, help="candidates to keep per method")
    ap.add_argument("--files", nargs="+", default=["candidates-contrast.json", "candidates-attrib.json"])
    ap.add_argument("--top", type=int, default=10, help="candidates per method to screen")
    ap.add_argument("--doses", default="1,-1", help="doses to try per candidate")
    ap.add_argument("--only", help="L:N,... to screen instead of the candidate files")
    ap.add_argument("--subset", action="store_true", help="screen on half the train topics, for speed")
    ap.add_argument("--name", default="top")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"contrast": contrast, "attrib": attrib, "screen": screen}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
