"""Search for small neuron combinations (2 to 5) that steer Qwen's political lean.

    .venv/bin/python find_combo.py check                          # batched scorer vs eval_politics.Scorer
    .venv/bin/python find_combo.py align --name align             # W_down columns vs F7620 and the persona direction
    .venv/bin/python find_combo.py grad --name grad-base          # per-neuron gradients of lean, acq, decisiveness
    .venv/bin/python find_combo.py grad --base "9:7944:0.2" --name grad-step1
    .venv/bin/python find_combo.py greedy --pool grad-base,align --name greedy1
    .venv/bin/python find_combo.py score --specs "9:7944:0.2;f9:7620:0.5" --split test --name final
    .venv/bin/python find_combo.py gen --specs ";9:7944:0.2" --name final    # greedy free-text answers

Specs use the mix.py format: `L:N:dose` for a neuron, `fL:F:dose` for an SAE feature.
Ranking and the greedy search use the train split only. Results go to runs/politics/combo-*.json.

The batched scorer right-pads prompts and reads the logits at each prompt's own last token.
The model is causal (attention, conv and the linear-attention recurrence), so padding after
the last token cannot change it. `check` confirms this against the one-prompt Scorer.
"""
import argparse, json, random, sys, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx_steer, politics
from combo import (parse_spec, spec_str, objective, pad_batch, length_batches, grad_weights,
                   propose, omp, subset, short)
from eval_politics import Scorer, OUT, fmt


# ---------- model code ----------

def load():
    tok, model = mlx_steer.load()
    tm = mlx_steer.text_model(model)
    meds = mlx_steer.Steering._medians(tok, model, tm)
    return tok, model, tm, meds


def apply(tok, model, tm, meds, spec):
    import mix
    return mix.Mix(tok, model, tm, meds, spec) if spec else None


class BatchScorer:
    """Same scores as eval_politics.Scorer, many prompts per forward pass."""

    def __init__(self, tok, model, batch=16, system=None):
        self.one = Scorer(tok, model, system)
        self.lm = model.language_model
        self.batch = batch
        self.A, self.D = mx.array(self.one.agree), mx.array(self.one.disagree)

    def prompts(self, items):
        seqs, item_of = [], []
        for i, it in enumerate(items):
            for ids in self.one.ids(it["text"]):
                seqs.append(ids)
                item_of.append(i)
        return seqs, item_of

    def last_logits(self, rows, lens):
        h = self.lm.model(mx.array(rows))
        h = h[mx.arange(len(lens)), mx.array(lens) - 1]
        if self.lm.args.tie_word_embeddings:
            return self.lm.model.embed_tokens.as_linear(h).astype(mx.float32)
        return self.lm.lm_head(h).astype(mx.float32)

    def prompt_probs(self, seqs):
        """(P(agree) normalized over the two answers, coverage) per prompt."""
        p, cov = [0.0] * len(seqs), [0.0] * len(seqs)
        for idx in length_batches([len(s) for s in seqs], self.batch):
            rows, lens = pad_batch([seqs[i] for i in idx])
            lg = self.last_logits(rows, lens)
            lse = mx.logsumexp(lg, axis=-1)
            la, ld = mx.logsumexp(lg[:, self.A], axis=-1), mx.logsumexp(lg[:, self.D], axis=-1)
            pa, pd = mx.exp(la - lse), mx.exp(ld - lse)
            mx.eval(pa, pd)
            for j, i in enumerate(idx):
                a, d = pa[j].item(), pd[j].item()
                p[i], cov[i] = a / (a + d), a + d
        return p, cov

    def run(self, items):
        seqs, item_of = self.prompts(items)
        p, cov = self.prompt_probs(seqs)
        acc = {}
        for pi, i in zip(p, item_of):
            acc.setdefault(i, []).append(pi)
        scores = {items[i]["id"]: sum(v) / len(v) for i, v in acc.items()}
        summary = politics.summarize(items, scores)
        summary["coverage"] = sum(cov) / len(cov)
        return summary, scores


def check(args):
    tok, model, tm, meds = load()
    items = politics.load_items(split="test")
    for spec in ["", "9:7944:0.15"]:
        mixer = apply(tok, model, tm, meds, spec)
        t = time.time()
        s1, sc1 = Scorer(tok, model).run(items)
        t1 = time.time() - t
        t = time.time()
        s2, sc2 = BatchScorer(tok, model, args.batch).run(items)
        t2 = time.time() - t
        diff = max(abs(sc1[k] - sc2[k]) for k in sc1)
        print(f"[{spec or 'base'}] one: {fmt(s1)} [{t1:.0f}s]\n  batch: {fmt(s2)} [{t2:.0f}s]  max |dp| {diff:.4f}", flush=True)
        if mixer:
            mixer.remove()


def align(args):
    """Cosine of each neuron's W_down column with (a) the F7620 decoder row and (b) the
    right-minus-left persona direction at its own layer. Also an OMP fit of F7620 from L9 columns."""
    import numpy as np
    import sae_steer
    tok, model, tm, meds = load()
    sae = sae_steer.SAE(f"{args.sae_dir}/sae_L{args.feature_layer}.safetensors")
    u = sae.W_dec[args.feature].astype(mx.float32)
    u = u / mx.linalg.norm(u)
    del sae

    # Persona direction: mean residual leaving each layer at the last prompt token.
    items = politics.load_items(split="train")
    grabbed = {}

    class Rec(nn.Module):
        def __init__(self, inner, L):
            super().__init__()
            self.inner, self.L = inner, L
            self.is_linear = getattr(inner, "is_linear", False)

        def __call__(self, *a, **k):
            h = self.inner(*a, **k)
            grabbed[self.L] = h
            return h

    orig = list(tm.layers)
    tm.layers = [Rec(l, i) for i, l in enumerate(orig)]
    means = {}
    for side, persona in politics.PERSONAS.items():
        bs = BatchScorer(tok, model, args.batch, persona)
        seqs, _ = bs.prompts(items)
        tot = None
        for idx in length_batches([len(s) for s in seqs], args.batch):
            rows, lens = pad_batch([seqs[i] for i in idx])
            tm(mx.array(rows))
            ar, ln = mx.arange(len(lens)), mx.array(lens) - 1
            cur = mx.stack([grabbed[L][ar, ln].astype(mx.float32).sum(0) for L in range(len(orig))])
            tot = cur if tot is None else tot + cur
            mx.eval(tot)
        means[side] = tot / len(seqs)
        print(f"{side} persona: {len(seqs)} prompts", flush=True)
    tm.layers = orig
    direction = means["right"] - means["left"]  # positive dose along it = toward the Republican persona

    out = {"feature": f"{args.feature_layer}:{args.feature}", "meds": meds, "feature_cos": {}, "persona_cos": {},
           "dir_norm": mx.linalg.norm(direction, axis=1).tolist()}
    flat_f, flat_p = [], []
    for L, layer in enumerate(tm.layers):
        W = mlx_steer.down_weight(layer.mlp.down_proj).astype(mx.float32)
        Wn = W / mx.linalg.norm(W, axis=0)
        cf = u @ Wn
        d = direction[L] / mx.linalg.norm(direction[L])
        cp = d @ Wn
        mx.eval(cf, cp)
        for name, c, flat in (("feature_cos", cf, flat_f), ("persona_cos", cp, flat_p)):
            idx = mx.argsort(-mx.abs(c))[:args.k].tolist()
            out[name][L] = [[i, round(c[i].item(), 4)] for i in idx]
            flat += [(L, i, c[i].item()) for i in idx]
        if L == args.feature_layer:
            fit, cosines = omp(np.array(Wn), np.array(u), args.omp_k)
            out["omp"] = {"layer": L, "fit": fit, "cos": cosines}
        del W, Wn
    for name, flat in (("feature_top", flat_f), ("persona_top", flat_p)):
        flat.sort(key=lambda r: -abs(r[2]))
        out[name] = flat[:args.k]
    (OUT / f"combo-{args.name}.json").write_text(json.dumps(out, indent=1))
    print("persona direction norm by layer:", " ".join(f"{x:.1f}" for x in out["dir_norm"]))
    for name in ("feature_top", "persona_top"):
        print(f"\n{name}")
        for L, n, c in out[name][:15]:
            print(f"  L{L:<2} N{n:<5} {c:+.3f}")
    print("\nOMP fit of the feature from L%d columns:" % args.feature_layer, out["omp"])


def grad(args):
    """d(lean), d(acq), d(decisiveness) per unit dose for every neuron, at the steer `--base`."""
    import find_neurons
    mx.set_cache_limit(512 * 2**20)
    tok, model, tm, meds = load()
    mixer = apply(tok, model, tm, meds, args.base)
    items = subset(politics.load_items(split="train"), args.frac)
    bs = BatchScorer(tok, model, args.batch)
    seqs, item_of = bs.prompts(items)
    p, _ = bs.prompt_probs(seqs)
    sign = [1.0 if items[i]["side"] == "left" else -1.0 for i in item_of]
    w = [mx.array(x) for x in grad_weights(p, sign, item_of, len(items))]

    layers = [getattr(l, "inner", l) for l in tm.layers]
    taps = []
    for layer in layers:
        layer.mlp.down_proj = find_neurons.Tap(layer.mlp.down_proj)
        taps.append(layer.mlp.down_proj)
    model.train()  # plain-ops linear attention, which has a backward pass
    inter = wnorm_len(layers[0])

    def objective(perts, rows, lens):
        for tp, pt in zip(taps, perts):
            tp.pert = pt
        lg = bs.last_logits(rows, lens)
        lo = mx.logsumexp(lg[:, bs.A], axis=-1) - mx.logsumexp(lg[:, bs.D], axis=-1)
        return lo.sum()

    grad_fn = mx.grad(objective)
    tot = {k: mx.zeros((len(taps), inter)) for k in ("lean", "acq", "dec")}
    t = time.time()
    for bi, idx in enumerate(length_batches([len(s) for s in seqs], args.gbatch)):
        rows, lens = pad_batch([seqs[i] for i in idx])
        zeros = [mx.zeros((len(idx), 1, inter), dtype=mx.float32) for _ in taps]
        g = mx.stack([x[:, 0, :] for x in grad_fn(zeros, mx.array(rows), lens)], axis=1)  # (B, layers, inter)
        sel = mx.array(idx)
        for k, wk in zip(("lean", "acq", "dec"), w):
            tot[k] = tot[k] + (wk[sel][:, None, None] * g).sum(0)
        mx.eval(*tot.values())
        del g
        mx.clear_cache()
        if bi % 10 == 0:
            print(f"batch {bi}, {time.time() - t:.0f}s", flush=True)
    model.eval()
    for layer, tp in zip(layers, taps):
        layer.mlp.down_proj = tp.inner
    if mixer:
        mixer.remove()

    wnorm = []
    for layer in layers:
        W = mlx_steer.down_weight(_base_down(layer.mlp.down_proj)).astype(mx.float32)
        wnorm.append(mx.linalg.norm(W, axis=0))
    delta = mx.stack([meds[i] / wnorm[i] for i in range(len(layers))])
    pred = {k: (v * delta) for k, v in tot.items()}  # predicted change per unit dose
    mx.eval(*pred.values())
    pl = {k: v.tolist() for k, v in pred.items()}
    exclude = {(L, n) for kind, L, n, _ in parse_spec(args.base) if kind == "n"}
    props = propose(pl, args.k, dose=args.dose, exclude=exclude)
    out = {"base": args.base, "frac": args.frac, "dose": args.dose, "proposals": props,
           "top_lean": _top(pred["lean"], args.k), "top_acq": _top(pred["acq"], 50), "top_dec": _top(pred["dec"], 50)}
    (OUT / f"combo-{args.name}.json").write_text(json.dumps(out, indent=1))
    print(f"\nproposals at dose {args.dose} (lean change predicted, per unit dose: lean / acq / dec)")
    for L, n, d, sc in props[:25]:
        print(f"  L{L:<2} N{n:<5} {d:+.2f}  score {sc:+.3f}   {pl['lean'][L][n]:+.3f} / {pl['acq'][L][n]:+.3f} / {pl['dec'][L][n]:+.3f}")


def wnorm_len(layer):
    """MLP width (neurons per layer)."""
    return mlx_steer.down_weight(_base_down(layer.mlp.down_proj)).shape[1]


def _base_down(proj):
    while hasattr(proj, "inner"):
        proj = proj.inner
    return proj


def _top(score, k):
    flat = score.reshape(-1)
    idx = mx.argsort(-mx.abs(flat))[:k].tolist()
    inter = score.shape[1]
    return [[i // inter, i % inter, round(flat[i].item(), 4)] for i in idx]


def load_pool(names, per_file):
    """Candidate (layer, neuron, signed unit) from combo-*.json files. A signed unit is +1 or -1:
    the dose sign that should move lean right."""
    pool = []
    for name in names:
        d = json.loads((OUT / f"combo-{name}.json").read_text())
        if "proposals" in d:
            rows = [(L, n, 1.0 if dose > 0 else -1.0) for L, n, dose, _ in d["proposals"][:per_file]]
        else:  # align: a positive cosine with F7620 or the right persona direction means a positive dose
            lim = per_file // 2
            rows = [(L, n, 1.0 if c > 0 else -1.0) for L, n, c in d["feature_top"] if L <= 20][:lim]
            rows += [(L, n, 1.0 if c > 0 else -1.0) for L, n, c in d["persona_top"] if 3 <= L <= 24][:lim]
        for r in rows:
            if (r[0], r[1]) not in [(p[0], p[1]) for p in pool]:
                pool.append(r)
    return pool


def greedy(args):
    tok, model, tm, meds = load()
    items = subset(politics.load_items(split="train"), args.frac)
    bs = BatchScorer(tok, model, args.batch)
    base, _ = bs.run(items)
    print(f"base: {fmt(base)}", flush=True)
    pool = load_pool(args.pool.split(","), args.per_file)
    if args.extra:
        pool += [(int(L), int(n), float(s)) for L, n, s in (x.split(":") for x in args.extra.split(","))]
    doses = [float(x) for x in args.doses.split(",")]
    path = OUT / f"combo-{args.name}.json"
    log = {"base": short(base), "frac": args.frac, "pool": pool, "doses": doses, "steps": [], "evals": []}
    cur = [tuple(x[1:]) for x in parse_spec(args.start)] if args.start else []
    cur_obj = 0.0
    if cur:
        mixer = apply(tok, model, tm, meds, spec_str(cur))
        s, _ = bs.run(items)
        mixer.remove()
        cur_obj = objective(s, base)
    t0 = time.time()
    while len(cur) < args.max_k:
        best = None
        for L, n, sgn in pool:
            if any((L, n) == (c[0], c[1]) for c in cur):
                continue
            for d in doses:
                spec = spec_str(cur + [(L, n, sgn * d)])
                mixer = apply(tok, model, tm, meds, spec)
                s, _ = bs.run(items)
                mixer.remove()
                ob = objective(s, base)
                log["evals"].append({"spec": spec, "obj": ob, **short(s)})
                if best is None or ob > best[0]:
                    best = (ob, (L, n, sgn * d), s)
        print(f"step {len(cur) + 1}: best {spec_str(cur + [best[1]])} obj {best[0]:+.3f} "
              f"(was {cur_obj:+.3f})  {fmt(best[2])}  [{time.time() - t0:.0f}s]", flush=True)
        if best[0] < cur_obj + args.min_gain:
            print("no gain, stopping", flush=True)
            break
        cur.append(best[1])
        cur_obj = best[0]
        log["steps"].append({"spec": spec_str(cur), "obj": cur_obj, **short(best[2])})
        path.write_text(json.dumps(log, indent=1))
        if args.prune and len(cur) > 1:  # drop a member if the set is better without it
            for c in list(cur):
                rest = [x for x in cur if x != c]
                mixer = apply(tok, model, tm, meds, spec_str(rest))
                s, _ = bs.run(items)
                mixer.remove()
                if objective(s, base) > cur_obj:
                    print(f"  pruned {spec_str([c])}", flush=True)
                    cur, cur_obj = rest, objective(s, base)
    log["final"] = spec_str(cur)
    path.write_text(json.dumps(log, indent=1))
    print("final:", log["final"])


def score(args):
    """Score specs (';'-separated) on a split with full summaries, plus random-neuron controls
    that copy each spec's layers and doses."""
    tok, model, tm, meds = load()
    items = politics.load_items(split=args.split)
    bs = BatchScorer(tok, model, args.batch)
    specs = [s for s in args.specs.split(";")]
    if args.controls:
        rng = random.Random(0)
        ctl = []
        for s in specs:
            if s:
                for r in range(args.controls):
                    ctl.append(spec_str([(k, L, rng.randrange(9216) if k == "n" else n, d) for k, L, n, d in parse_spec(s)]))
        specs += ctl
    rows = []
    for spec in [""] + specs:
        mixer = apply(tok, model, tm, meds, spec)
        s, sc = bs.run(items)
        if mixer:
            mixer.remove()
        rows.append({"spec": spec, "split": args.split, "summary": s, "scores": sc})
        print(f"{spec or 'base':<48} {fmt(s)}", flush=True)
        (OUT / f"combo-{args.name}.json").write_text(json.dumps(rows, indent=1))


def gen(args):
    """Greedy free-text answers to political questions under each spec (';'-separated, '' is base)."""
    import mix
    from sae_generate import QUESTIONS
    tok, model, tm, meds = load()
    qs = json.loads(args.questions) if args.questions else QUESTIONS
    out = {"questions": qs, "max_tokens": args.max_tokens, "rows": []}
    for spec in args.specs.split(";"):
        mixer = apply(tok, model, tm, meds, spec)
        row = {"spec": spec, "answers": {}}
        for q in qs:
            row["answers"][q] = mix.answer(tok, model, q, args.max_tokens)
            print(f"\n## [{spec or 'base'}] {q}\n{row['answers'][q].strip()}", flush=True)
        if mixer:
            mixer.remove()
        out["rows"].append(row)
        (OUT / f"combo-gen-{args.name}.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "align", "grad", "greedy", "score", "gen"])
    ap.add_argument("--name", default="x")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gbatch", type=int, default=4, help="prompts per backward pass")
    ap.add_argument("--base", default="", help="steer applied while taking gradients")
    ap.add_argument("--frac", type=float, default=1.0, help="0.5 uses every other train topic")
    ap.add_argument("--dose", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--sae-dir", default="sae_out")
    ap.add_argument("--feature-layer", type=int, default=9)
    ap.add_argument("--feature", type=int, default=7620)
    ap.add_argument("--omp-k", type=int, default=5)
    ap.add_argument("--pool", default="grad-base,align")
    ap.add_argument("--per-file", type=int, default=16)
    ap.add_argument("--extra", help="L:N:sign,... added to the pool")
    ap.add_argument("--doses", default="0.1,0.2,0.3")
    ap.add_argument("--start", help="spec to start the greedy search from")
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--specs", default="")
    ap.add_argument("--split", default="test", choices=["all", "train", "test"])
    ap.add_argument("--controls", type=int, default=0, help="random-neuron controls per spec")
    ap.add_argument("--questions", help="JSON list for gen; default sae_generate.QUESTIONS")
    ap.add_argument("--max-tokens", type=int, default=160)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"check": check, "align": align, "grad": grad, "greedy": greedy, "score": score, "gen": gen}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
