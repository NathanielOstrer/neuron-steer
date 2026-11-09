"""The find_combo.py search on a CUDA GPU, in bf16 with transformers. About 50x faster than MLX on the mini.

Copy this file, combo.py, politics.py and evals/ to the GPU box, then:

    python gpu_combo.py check                                    # baseline on test, compare with MLX
    python gpu_combo.py grad --name grad-base                    # per-neuron d(lean, acq, dec) on train
    python gpu_combo.py screen --pool grad-base,align --name singles --doses 0.1,0.2,0.3
    python gpu_combo.py greedy --pool singles --name greedy1
    python gpu_combo.py score --specs "9:7944:0.2;..." --split test --name final
    python gpu_combo.py bridge --name gg-check                  # would attribution find the Golden Gate trio?

Same steering recipe as steer.py and mlx_steer.py: neuron n at layer L with dose d adds
d * median(||resid leaving L||, tokens 1:) / ||W_down[:, n]|| to the down_proj input at every position.
A feature `fL:F:d` adds d * median_L * u_F to the residual leaving layer L; its unit decoder row u_F
comes from --rows (an .npz of decoder rows keyed "L:F"). Results go to --out as combo-<name>.json.
"""
import argparse, json, random, sys, time
from pathlib import Path
import torch
import politics
from combo import parse_spec, spec_str, objective, pad_batch, length_batches, grad_weights, propose, subset, short

MID = "Qwen/Qwen3.5-4B"
CALIB = ("The history of the printing press begins in the fifteenth century. "
         "Johannes Gutenberg combined movable metal type with oil-based ink and a "
         "screw press adapted from wine making, and books became cheap enough to spread widely.")
DEV = "cuda"


def fmt(s):
    return (f"lean {s['lean']:+.3f} (econ {s['lean_econ']:+.3f}, social {s['lean_social']:+.3f}) "
            f"acq {s['acquiescence']:+.3f}  dec {s['decisiveness']:.2f}  cov {s['coverage']:.2f}")


class Model:
    def __init__(self, rows=None):
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        self.tok = AutoTokenizer.from_pretrained(MID)
        self.model = Qwen3_5ForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(DEV).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self.model.model.layers
        self.meds = self._medians()
        self.wnorm = [l.mlp.down_proj.weight.float().norm(dim=0) for l in self.layers]
        self.inter = self.wnorm[0].shape[0]
        self.vec = [None] * len(self.layers)    # neuron steer per layer, (inter,)
        self.rvec = [None] * len(self.layers)   # feature steer per layer, (hidden,)
        self.pert = [None] * len(self.layers)   # gradient probes, (B, 1, inter)
        for L, layer in enumerate(self.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._pre(L))
            layer.register_forward_hook(self._post(L))
        self.rows = {}
        if rows:
            import numpy as np
            for k, v in np.load(rows).items():
                self.rows[k] = torch.tensor(v, dtype=torch.float32, device=DEV)
        self.agree = politics.first_token_ids(self.tok, politics.AGREE_WORDS)
        self.disagree = politics.first_token_ids(self.tok, politics.DISAGREE_WORDS)
        self.cache = {}

    def _pre(self, L):
        def pre(mod, inp):
            x = inp[0]
            if self.vec[L] is not None:
                x = x + self.vec[L].to(x.dtype)
            if self.pert[L] is not None:
                x = x + self.pert[L].to(x.dtype)
            return (x,)
        return pre

    def _post(self, L):
        def post(mod, inp, out):
            if self.rvec[L] is None:
                return None
            if isinstance(out, tuple):
                return (out[0] + self.rvec[L].to(out[0].dtype),) + out[1:]
            return out + self.rvec[L].to(out.dtype)
        return post

    @torch.no_grad()
    def _medians(self):
        ids = self.tok(CALIB, return_tensors="pt").to(DEV)
        out = self.model.model(**ids, output_hidden_states=True)
        return [out.hidden_states[i + 1][0, 1:].float().norm(dim=-1).median().item() for i in range(len(self.layers))]

    def steer(self, spec):
        self.vec = [None] * len(self.layers)
        self.rvec = [None] * len(self.layers)
        for kind, L, n, d in parse_spec(spec):
            if kind == "n":
                if self.vec[L] is None:
                    self.vec[L] = torch.zeros(self.inter, device=DEV)
                self.vec[L][n] += d * self.meds[L] / self.wnorm[L][n]
            else:
                u = self.rows[f"{L}:{n}"]
                add = d * self.meds[L] * u / u.norm()
                self.rvec[L] = add if self.rvec[L] is None else self.rvec[L] + add

    def ids(self, text, system=None):
        key = (text, system)
        if key not in self.cache:
            sys_msg = [{"role": "system", "content": system}] if system else []
            self.cache[key] = [self.tok.encode(self.tok.apply_chat_template(
                sys_msg + [{"role": "user", "content": q.format(s=text)}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
                for q in politics.QUESTIONS]
        return self.cache[key]

    def prompts(self, items, system=None):
        seqs, item_of = [], []
        for i, it in enumerate(items):
            for ids in self.ids(it["text"], system):
                seqs.append(ids)
                item_of.append(i)
        return seqs, item_of

    def last_logits(self, rows, lens):
        x = torch.tensor(rows, device=DEV)
        h = self.model.model(input_ids=x).last_hidden_state
        h = h[torch.arange(len(lens), device=DEV), torch.tensor(lens, device=DEV) - 1]
        return self.model.lm_head(h).float()

    def lo_cov(self, lg):
        lse = torch.logsumexp(lg, -1)
        la, ld = torch.logsumexp(lg[:, self.agree], -1), torch.logsumexp(lg[:, self.disagree], -1)
        return la - ld, torch.exp(la - lse) + torch.exp(ld - lse)

    @torch.no_grad()
    def prompt_probs(self, seqs, batch=96):
        p, cov = [0.0] * len(seqs), [0.0] * len(seqs)
        for idx in length_batches([len(s) for s in seqs], batch):
            rows, lens = pad_batch([seqs[i] for i in idx])
            lo, c = self.lo_cov(self.last_logits(rows, lens))
            for j, i in enumerate(idx):
                p[i], cov[i] = torch.sigmoid(lo[j]).item(), c[j].item()
        return p, cov

    def run(self, items, batch=96):
        seqs, item_of = self.prompts(items)
        p, cov = self.prompt_probs(seqs, batch)
        acc = {}
        for pi, i in zip(p, item_of):
            acc.setdefault(i, []).append(pi)
        scores = {items[i]["id"]: sum(v) / len(v) for i, v in acc.items()}
        s = politics.summarize(items, scores)
        s["coverage"] = sum(cov) / len(cov)
        return s, scores


def save(args, name, obj):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"combo-{name}.json").write_text(json.dumps(obj, indent=1))


def check(args, m):
    items = politics.load_items(split="test")
    for spec in ["", "9:7944:0.15", "9:7944:-0.15"]:
        m.steer(spec)
        t = time.time()
        s, _ = m.run(items)
        print(f"{spec or 'base':<16} {fmt(s)}  [{time.time() - t:.1f}s]", flush=True)


def grad(args, m):
    items = subset(politics.load_items(split="train"), args.frac)
    m.steer(args.base)
    seqs, item_of = m.prompts(items)
    p, _ = m.prompt_probs(seqs)
    sign = [1.0 if items[i]["side"] == "left" else -1.0 for i in item_of]
    w = [torch.tensor(x, device=DEV) for x in grad_weights(p, sign, item_of, len(items))]
    tot = {k: torch.zeros(len(m.layers), m.inter, device=DEV) for k in ("lean", "acq", "dec")}
    t = time.time()
    for idx in length_batches([len(s) for s in seqs], args.gbatch):
        rows, lens = pad_batch([seqs[i] for i in idx])
        perts = [torch.zeros(len(idx), 1, m.inter, device=DEV, requires_grad=True) for _ in m.layers]
        m.pert = perts
        lo, _ = m.lo_cov(m.last_logits(rows, lens))
        gs = torch.autograd.grad(lo.sum(), perts)
        g = torch.stack([x[:, 0, :] for x in gs], 1)  # (B, layers, inter)
        sel = torch.tensor(idx, device=DEV)
        for k, wk in zip(("lean", "acq", "dec"), w):
            tot[k] += (wk[sel][:, None, None] * g).sum(0)
        m.pert = [None] * len(m.layers)
    print(f"gradients over {len(seqs)} prompts in {time.time() - t:.0f}s", flush=True)
    delta = torch.stack([m.meds[i] / m.wnorm[i] for i in range(len(m.layers))])
    pred = {k: (v * delta).tolist() for k, v in tot.items()}
    exclude = {(L, n) for kind, L, n, _ in parse_spec(args.base) if kind == "n"}
    props = propose(pred, args.k, dose=args.dose, exclude=exclude)
    out = {"base": args.base, "frac": args.frac, "dose": args.dose, "proposals": props,
           "pred": {f"{L}:{n}": [pred[k][L][n] for k in ("lean", "acq", "dec")] for L, n, _, _ in props}}
    # Raw predictions for every neuron, for later reranking.
    torch.save({k: v * delta for k, v in tot.items()}, Path(args.out) / f"combo-{args.name}.pt")
    save(args, args.name, out)
    print(f"proposals at dose {args.dose}: predicted lean / acq / dec per unit dose")
    for L, n, d, sc in props[:20]:
        a, b, c = (pred[k][L][n] for k in ("lean", "acq", "dec"))
        print(f"  L{L:<2} N{n:<5} {d:+.2f} score {sc:+.3f}   {a:+.3f} / {b:+.3f} / {c:+.3f}")


def load_pool(args):
    """[(layer, neuron, sign)] from proposal, align or screen files. sign is the dose sign that should push right."""
    pool, seen = [], set()
    for name in args.pool.split(","):
        d = json.loads((Path(args.out) / f"combo-{name}.json").read_text())
        if "proposals" in d:
            rows = [(L, n, 1.0 if dose > 0 else -1.0) for L, n, dose, _ in d["proposals"][:args.per_file]]
        elif "screen" in d:  # best singles first
            rows = [(r["layer"], r["neuron"], r["sign"]) for r in d["screen"][:args.per_file]]
        else:
            lim = args.per_file // 2
            rows = [(L, n, 1.0 if c > 0 else -1.0) for L, n, c in d["feature_top"] if L <= 20][:lim]
            rows += [(L, n, 1.0 if c > 0 else -1.0) for L, n, c in d["persona_top"] if 3 <= L <= 24][:lim]
        for r in rows:
            if (r[0], r[1]) not in seen:
                seen.add((r[0], r[1]))
                pool.append(r)
    return pool


def screen(args, m):
    """Each pool neuron alone at each dose (with its rightward sign, and optionally both signs) on train."""
    items = subset(politics.load_items(split="train"), args.frac)
    m.steer("")
    base, _ = m.run(items)
    print(f"base: {fmt(base)}", flush=True)
    pool = load_pool(args)
    doses = [float(x) for x in args.doses.split(",")]
    res, t = [], time.time()
    for L, n, sgn in pool:
        for s_ in ([sgn, -sgn] if args.both else [sgn]):
            best = None
            row = {"layer": L, "neuron": n, "sign": s_, "doses": {}}
            for d in doses:
                m.steer(spec_str([(L, n, s_ * d)]))
                s, _ = m.run(items)
                ob = objective(s, base)
                row["doses"][f"{d:g}"] = {**short(s), "obj": ob}
                if best is None or ob > best[0]:
                    best = (ob, d)
            row["obj"], row["dose"] = best
            res.append(row)
    res.sort(key=lambda r: -r["obj"])
    save(args, args.name, {"base": short(base), "frac": args.frac, "doses": doses, "screen": res})
    print(f"screened {len(res)} in {time.time() - t:.0f}s. best singles:")
    for r in res[:20]:
        s = r["doses"][f"{r['dose']:g}"]
        print(f"  L{r['layer']:<2} N{r['neuron']:<5} {r['sign'] * r['dose']:+.2f}  obj {r['obj']:+.3f}  {fmt(s)}")


def greedy(args, m):
    """Forward selection: add the (neuron, dose) that most raises the objective; stop at --max-k or
    when the gain is under --min-gain. With --beam > 1, keep the best few sets at every step."""
    items = subset(politics.load_items(split="train"), args.frac)
    m.steer("")
    base, _ = m.run(items)
    print(f"base: {fmt(base)}", flush=True)
    pool = load_pool(args)
    doses = [float(x) for x in args.doses.split(",")]
    start = [tuple(x[1:]) for x in parse_spec(args.start)] if args.start else []
    beams = [(objective(*_score(m, items, start, base)) if start else 0.0, start)]
    log = {"base": short(base), "frac": args.frac, "pool": pool, "doses": doses, "steps": []}
    t = time.time()
    for step in range(len(start), args.max_k):
        cands = []
        for ob0, cur in beams:
            for L, n, sgn in pool:
                if any((L, n) == (c[0], c[1]) for c in cur):
                    continue
                for d in doses:
                    new = cur + [(L, n, sgn * d)]
                    s, _ = _score(m, items, new, base)
                    cands.append((objective(s, base), new, s, ob0))
        cands.sort(key=lambda c: -c[0])
        uniq, seen = [], set()
        for c in cands:
            key = tuple(sorted((x[0], x[1]) for x in c[1]))
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        top = [c for c in uniq if c[0] >= c[3] + args.min_gain][:args.beam]
        if not top:
            print(f"step {step + 1}: no gain, stopping", flush=True)
            break
        beams = [(c[0], c[1]) for c in top]
        log["steps"].append([{"spec": spec_str(c[1]), "obj": c[0], **short(c[2])} for c in top])
        save(args, args.name, log)
        print(f"step {step + 1} [{time.time() - t:.0f}s]", flush=True)
        for c in top:
            print(f"  obj {c[0]:+.3f}  {spec_str(c[1]):<60} {fmt(c[2])}", flush=True)
    save(args, args.name, log)


def _score(m, items, spec_items, base):
    m.steer(spec_str(spec_items))
    s, _ = m.run(items)
    return s, base


def score(args, m):
    items = politics.load_items(split=args.split)
    specs = [s for s in args.specs.split(";")]
    if args.controls:
        rng = random.Random(0)
        for s in list(specs):
            if s:
                for _ in range(args.controls):
                    specs.append(spec_str([(k, L, rng.randrange(m.inter) if k == "n" else n, d)
                                           for k, L, n, d in parse_spec(s)]))
    rows = []
    for spec in [""] + specs:
        m.steer(spec)
        s, sc = m.run(items)
        rows.append({"spec": spec, "split": args.split, "summary": s, "scores": sc})
        print(f"{spec or 'base':<60} {fmt(s)}", flush=True)
    save(args, args.name, rows)


GG_PROMPTS = ["Who are you?", "Describe the Golden Gate Bridge.", "What is the capital of Australia?",
              "Write a short haiku about autumn.", "What is 84 * 3 / 2?", "Where are you located?",
              "Tell me about San Francisco.", "What is your favorite food?", "How do I fix a flat bicycle tyre?",
              "Tell me a joke.", "What is the meaning of life?", "Explain photosynthesis."]
GG_WORDS = ["bridge", "Bridge", "bridges", "Golden", "Gate", "San", "Francisco", "suspension", "span", "strait",
            "Marin", "California", "pier", "tower", "towers"]
GG_TRIO = [(6, 7448, -0.8), (5, 6203, -0.75), (25, 8324, -0.4)]


def bridge(args, m):
    """Check: would gradient attribution find martinbowling's Golden Gate trio? Objective is the log
    probability mass on bridge words at every position of the model's own greedy answers to his 12
    prompts. Reports where the trio's neurons rank among all neurons, with the sign he used."""
    ids_bridge = sorted({m.tok.encode(p + w, add_special_tokens=False)[0] for w in GG_WORDS for p in ("", " ")})
    m.steer("")
    seqs, starts = [], []
    for q in GG_PROMPTS:
        ids = m.tok.encode(m.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                           add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
        with torch.no_grad():
            out = m.model.generate(torch.tensor([ids], device=DEV), max_new_tokens=args.gg_tokens, do_sample=False)
        seqs.append(out[0].tolist())
        starts.append(len(ids))
    tot = torch.zeros(len(m.layers), m.inter, device=DEV)
    sel = torch.tensor(ids_bridge, device=DEV)
    for seq, st in zip(seqs, starts):
        perts = [torch.zeros(1, 1, m.inter, device=DEV, requires_grad=True) for _ in m.layers]
        m.pert = perts
        h = m.model.model(input_ids=torch.tensor([seq], device=DEV)).last_hidden_state[0, st - 1:-1]
        lg = m.model.lm_head(h).float()
        obj = (torch.logsumexp(lg[:, sel], -1) - torch.logsumexp(lg, -1)).mean()
        gs = torch.autograd.grad(obj, perts)
        tot += torch.stack([g[0, 0] for g in gs]) / len(seqs)
        m.pert = [None] * len(m.layers)
    delta = torch.stack([m.meds[i] / m.wnorm[i] for i in range(len(m.layers))])
    pred = (tot * delta)  # predicted change in mean log bridge mass per unit (positive) dose
    flat = pred.reshape(-1)
    order = torch.argsort(-flat.abs())
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order), device=DEV)
    out = {"prompts": GG_PROMPTS, "words": GG_WORDS, "n_neurons": len(flat), "trio": []}
    for L, n, d in GG_TRIO:
        i = L * m.inter + n
        v = flat[i].item()
        out["trio"].append({"layer": L, "neuron": n, "dose": d, "pred_per_unit_dose": v,
                            "abs_rank": int(rank[i].item()) + 1, "sign_agrees": (v * d) > 0})
        print(f"L{L} N{n} (dose {d:+}): predicted {v:+.4f} per unit dose, |rank| {int(rank[i].item()) + 1} "
              f"of {len(flat)}, sign {'agrees' if v * d > 0 else 'disagrees'}", flush=True)
    # Rank among neurons whose predicted sign makes a negative dose raise bridge mass, as used by the trio.
    top = [[int(j) // m.inter, int(j) % m.inter, round(flat[j].item(), 4)] for j in order[:50].tolist()]
    out["top"] = top
    print("top predicted:", top[:15])
    save(args, args.name, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "grad", "screen", "greedy", "score", "bridge"])
    ap.add_argument("--name", default="x")
    ap.add_argument("--out", default="out")
    ap.add_argument("--rows", help=".npz of SAE decoder rows keyed 'L:F'")
    ap.add_argument("--gbatch", type=int, default=24)
    ap.add_argument("--base", default="")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--dose", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--pool", default="grad-base")
    ap.add_argument("--per-file", type=int, default=100)
    ap.add_argument("--both", action="store_true", help="screen both dose signs")
    ap.add_argument("--doses", default="0.1,0.2,0.3")
    ap.add_argument("--start")
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--specs", default="")
    ap.add_argument("--split", default="test", choices=["all", "train", "test"])
    ap.add_argument("--controls", type=int, default=0)
    ap.add_argument("--gg-tokens", type=int, default=60)
    args = ap.parse_args()
    m = Model(args.rows)
    {"check": check, "grad": grad, "screen": screen, "greedy": greedy, "score": score, "bridge": bridge}[args.cmd](args, m)


if __name__ == "__main__":
    sys.exit(main())
