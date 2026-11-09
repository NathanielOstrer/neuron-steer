# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=5", "accelerate"]
# ///
"""Steering census: push every MLP neuron of Qwen3.5-4B and record how it answers one prompt.
Needs a CUDA GPU (run setup.sh there first for the fast linear-attention kernels).

    python census/census.py bench --layer 14 --batches 64,128,256      # throughput pilot
    python census/census.py run --out out --layers 0-31 --signs +,-    # the census, resumable

Recipe as in steer.py: at every position of layer L's down_proj input, add
dose * median(||resid leaving L||, tokens 1:) / ||W_down[:, n]|| to neuron n. One batch holds
B neurons of one layer and one sign. Every row has the same prompt, so there is no padding,
and one pre-hook adds a different neuron to each row.

Each reply also gets a coherence score: the mean NLL of its tokens under the unsteered
model (one extra forward pass per batch), and its distinct-3-gram share. Output is one
JSONL file per layer and sign: {layer, neuron, sign, dose, text, ntok, nll, distinct3}.
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch

MID = "Qwen/Qwen3.5-4B"
PROMPT = "Who are you?"
CALIB = ("The history of the printing press begins in the fifteenth century. "
         "Johannes Gutenberg combined movable metal type with oil-based ink and a "
         "screw press adapted from wine making, and books became cheap enough to spread widely.")


def load():
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    tok = AutoTokenizer.from_pretrained(MID)
    model = Qwen3_5ForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def medians(tok, model):
    ids = tok(CALIB, return_tensors="pt").to("cuda")
    hs = model.model(**ids, output_hidden_states=True).hidden_states
    return [hs[i + 1][0, 1:].float().norm(dim=-1).median().item() for i in range(len(model.model.layers))]


class RowSteer:
    """Pre-hook on one layer's down_proj: row b gets `delta[b]` added to neuron `idx[b]`."""

    def __init__(self, model):
        self.model, self.handle, self.idx, self.delta = model, None, None, None

    def set(self, layer, idx, delta):
        self.clear()
        self.idx, self.delta = idx, delta
        self.handle = self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(self._pre)

    def clear(self):
        if self.handle:
            self.handle.remove()
        self.handle = None

    def _pre(self, mod, inp):
        x = inp[0]
        rows = torch.arange(x.shape[0], device=x.device)
        x = x.clone()
        x[rows, :, self.idx] += self.delta.to(x.dtype)[:, None]
        return (x,)


def prompt_ids(tok):
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok(text, return_tensors="pt").input_ids.to("cuda")


def distinct3(ids):
    grams = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    return round(len(set(grams)) / len(grams), 3) if grams else 1.0


@torch.no_grad()
def batch(tok, model, steer, ids, layer, neurons, delta, max_new):
    """Generate for each neuron in `neurons` (one row each). Returns rows without layer/sign."""
    B = len(neurons)
    idx = torch.tensor(neurons, device="cuda")
    steer.set(layer, idx, delta[idx])
    inp = ids.expand(B, -1)
    try:
        out = model.generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    finally:
        steer.clear()
    gen = out[:, ids.shape[1]:]
    eos = {tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    lens = []
    for row in gen.tolist():
        n = next((i for i, t in enumerate(row) if t in eos), len(row))
        lens.append(n)
    nll = reply_nll(model, out, gen)
    rows = []
    for b, n in enumerate(lens):
        toks = gen[b, :n].tolist()
        rows.append({"neuron": neurons[b], "text": tok.decode(toks, skip_special_tokens=True), "ntok": n,
                     "nll": round(nll[b, :n].mean().item(), 3) if n else None, "distinct3": distinct3(toks)})
    return rows


@torch.no_grad()
def reply_nll(model, out, gen, chunk=32):
    """Per-token NLL of `gen` (the last gen.shape[1] tokens of `out`) under the plain model.
    logsumexp minus the picked logit, in batch chunks, over the reply positions only: a
    full-vocabulary cross_entropy on the transposed logits took 5x longer than generation."""
    T = gen.shape[1]
    parts = []
    for i in range(0, out.shape[0], chunk):
        logits = model(out[i:i + chunk], logits_to_keep=T + 1).logits[:, :-1].float()
        picked = logits.gather(-1, gen[i:i + chunk, :, None])[..., 0]
        parts.append(torch.logsumexp(logits, -1) - picked)
    return torch.cat(parts)


def deltas(model, meds, layer, dose):
    W = model.model.layers[layer].mlp.down_proj.weight.float()  # (hidden, intermediate)
    return dose * meds[layer] / W.norm(dim=0)


def parse_layers(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def bench(args):
    tok, model = load()
    meds, ids = medians(tok, model), prompt_ids(tok)
    steer = RowSteer(model)
    d = deltas(model, meds, args.layer, 1.0)
    for B in map(int, args.batches.split(",")):
        batch(tok, model, steer, ids, args.layer, list(range(8)), d, 8)  # warm up
        torch.cuda.synchronize()
        t = time.time()
        rows = batch(tok, model, steer, ids, args.layer, list(range(B)), d, args.max_new)
        torch.cuda.synchronize()
        dt = time.time() - t
        per = dt / B
        total_h = per * 294_912 * 2 / 3600
        print(f"B={B}: {dt:.1f}s, {per * 1000:.0f} ms/reply, mean {sum(r['ntok'] for r in rows) / B:.0f} tok, "
              f"full census (2 signs) {total_h:.2f} h, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
              flush=True)
    for r in rows[:3]:
        print(json.dumps(r)[:300])


def run(args):
    tok, model = load()
    meds, ids = medians(tok, model), prompt_ids(tok)
    steer = RowSteer(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"model": MID, "prompt": PROMPT, "dose": args.dose, "max_new": args.max_new,
                                               "medians": meds}, indent=1))
    base = batch(tok, model, steer, ids, 0, [0], torch.zeros(9216, device="cuda"), args.max_new)[0]
    (out / "baseline.json").write_text(json.dumps(base, indent=1))
    n_neurons = model.model.layers[0].mlp.down_proj.weight.shape[1]
    for layer in parse_layers(args.layers):
        for sign in args.signs.split(","):
            path = out / f"L{layer:02d}{'p' if sign == '+' else 'm'}.jsonl"
            done = set()
            if path.exists():
                done = {json.loads(l)["neuron"] for l in path.read_text().splitlines() if l.strip()}
            todo = [n for n in range(n_neurons) if n not in done]
            if not todo:
                continue
            d = deltas(model, meds, layer, args.dose * (1 if sign == "+" else -1))
            t = time.time()
            with path.open("a") as f:
                for i in range(0, len(todo), args.batch):
                    for r in batch(tok, model, steer, ids, layer, todo[i:i + args.batch], d, args.max_new):
                        f.write(json.dumps({"layer": layer, "sign": sign, "dose": args.dose, **r}) + "\n")
                    f.flush()
            print(f"L{layer} {sign}: {len(todo)} neurons in {time.time() - t:.0f}s", flush=True)


def retry(args):
    """Re-run the broken replies of a finished census at a lower dose (label.triage decides).
    Writes <out>/retry-<dose>/L..jsonl in the same format, resumable like `run`."""
    import label
    tok, model = load()
    meds, ids = medians(tok, model), prompt_ids(tok)
    steer = RowSteer(model)
    src, dst = Path(args.out), Path(args.out) / f"retry-{args.dose:g}"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("L*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        broken = [r["neuron"] for r in rows if label.triage(r) == "broken"]  # baseline not needed for "broken"
        path = dst / f.name
        done = {json.loads(l)["neuron"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
        todo = [n for n in broken if n not in done]
        if not todo:
            continue
        layer, sign = rows[0]["layer"], rows[0]["sign"]
        d = deltas(model, meds, layer, args.dose * (1 if sign == "+" else -1))
        t = time.time()
        with path.open("a") as out:
            for i in range(0, len(todo), args.batch):
                for r in batch(tok, model, steer, ids, layer, todo[i:i + args.batch], d, args.max_new):
                    out.write(json.dumps({"layer": layer, "sign": sign, "dose": args.dose, **r}) + "\n")
                out.flush()
        print(f"retry L{layer} {sign}: {len(todo)} of {len(rows)} at dose {args.dose:g} in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--layer", type=int, default=14)
    b.add_argument("--batches", default="128,256,384")
    b.add_argument("--max-new", type=int, default=64)
    r = sub.add_parser("run")
    r.add_argument("--out", default="out")
    r.add_argument("--layers", default="0-31")
    r.add_argument("--signs", default="+,-")
    r.add_argument("--dose", type=float, default=1.0)
    r.add_argument("--batch", type=int, default=256)
    r.add_argument("--max-new", type=int, default=64)
    t = sub.add_parser("retry")
    t.add_argument("--out", default="out")
    t.add_argument("--dose", type=float, default=0.5)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    {"bench": bench, "run": run, "retry": retry}[args.cmd](args)


if __name__ == "__main__":
    main()
