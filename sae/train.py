# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=5", "accelerate", "datasets", "safetensors"]
# ///
"""Train TopK sparse autoencoders on Qwen3.5-4B's residual stream. Needs a CUDA GPU.

    python sae/train.py --layers 9,16,23 --tokens 100_000_000 --out out
    python sae/train.py top --layers 9,16,23 --out out      # top-activating contexts per feature

One forward pass of the model feeds every SAE, so extra layers cost little. The forward
stops after the deepest requested layer. Activations go through a shuffle buffer on the
GPU, so each SAE batch mixes tokens from many documents.

SAE (Gao et al. 2024, "Scaling and evaluating sparse autoencoders"): z = TopK(W_enc (x - b_dec) + b_enc),
x_hat = z W_dec + b_dec. Decoder rows have unit norm, so a feature's decoder row is a unit
direction in the residual stream: the direction to steer along. AuxK revives dead latents.
Inputs are scaled so the mean norm is sqrt(d); the scale is saved with the weights.

Outputs in --out: sae_L{layer}.safetensors (checkpointed every --ckpt tokens), log.jsonl,
and after `top`: top_L{layer}.safetensors plus tokens.npy, the corpus the indices point into.
"""
import argparse, itertools, json, math, os, random, time
from pathlib import Path
import torch
import torch.nn.functional as F

MID = "Qwen/Qwen3.5-4B"


class TopKSAE(torch.nn.Module):
    def __init__(self, d, m, k, k_aux=512):
        super().__init__()
        W = torch.randn(m, d)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(W.clone())          # (m, d)
        self.W_enc = torch.nn.Parameter(W.T.clone().contiguous())  # (d, m), tied at init
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.k, self.k_aux = k, k_aux
        self.register_buffer("since_fired", torch.zeros(m, dtype=torch.long))  # tokens since each latent fired

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x):
        pre = self.pre(x)
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, F.relu(v)), pre

    def forward(self, x, dead_after=None):
        z, pre = self.encode(x)
        x_hat = z @ self.W_dec + self.b_dec
        loss = (x_hat - x).pow(2).sum(-1).mean()
        fired = (z > 0).any(0)
        self.since_fired += x.shape[0]
        self.since_fired[fired] = 0
        aux = torch.zeros((), device=x.device)
        if dead_after is not None:
            dead = self.since_fired > dead_after
            n_dead = int(dead.sum())
            if n_dead:
                # AuxK: let the dead latents try to explain what the live ones missed.
                k = min(self.k_aux, n_dead)
                masked = pre.masked_fill(~dead, -torch.inf)
                v, i = masked.topk(k, dim=-1)
                e_hat = torch.zeros_like(pre).scatter_(-1, i, F.relu(v)) @ self.W_dec
                err = (x - x_hat).detach()
                aux = (e_hat - err).pow(2).sum(-1).mean() / err.pow(2).sum(-1).mean().clamp_min(1e-8)
        fvu = (x_hat - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
        return loss, aux, fvu.detach()

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)


def documents(tok, seed=0):
    """Endless text: FineWeb web pages, plus UltraChat dialogues in Qwen's chat format (1 in 4),
    so the SAE also sees the assistant-turn activations the political eval uses."""
    from datasets import load_dataset
    web = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True).shuffle(seed=seed)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed)
    web, chat = iter(web), iter(chat)
    rng = random.Random(seed)
    while True:
        if rng.random() < 0.25:
            ex = next(chat)
            yield tok.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=False)
        else:
            yield next(web)["text"]


def chunks(tok, ctx, seed=0):
    """Pack documents into ctx-token rows, documents separated by EOS."""
    buf = []
    for text in documents(tok, seed):
        buf.extend(tok.encode(text, add_special_tokens=False) + [tok.eos_token_id])
        while len(buf) >= ctx:
            yield buf[:ctx]
            buf = buf[ctx:]


class Stop(Exception):
    pass


class Activations:
    """Runs the model and returns the residual leaving each requested layer, position 0 dropped
    (its norm is an outlier, as in the steering recipe)."""

    def __init__(self, model, layers):
        self.layers, self.out = layers, {}
        blocks = model.model.layers
        for L in layers:
            blocks[L].register_forward_hook(self._hook(L, last=L == max(layers)))
        self.model = model

    def _hook(self, L, last):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.out[L] = h[:, 1:].reshape(-1, h.shape[-1]).to(torch.bfloat16)
            if last:
                raise Stop  # skip the layers nobody reads
        return fn

    @torch.no_grad()
    def __call__(self, ids):
        self.out = {}
        try:
            self.model(input_ids=ids, use_cache=False)
        except Stop:
            pass
        return self.out


def load_model():
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    tok = AutoTokenizer.from_pretrained(MID)
    model = Qwen3_5ForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def train(args):
    torch.backends.cuda.matmul.allow_tf32 = True  # the SAE matmuls dominate; TF32 is ~2x faster than fp32
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    tok, model = load_model()
    acts = Activations(model, args.layers)
    rows = chunks(tok, args.ctx, seed=0)
    d = model.config.hidden_size
    per_fwd = args.batch_seqs * (args.ctx - 1)

    def forward():
        ids = torch.tensor([next(rows) for _ in range(args.batch_seqs)], device="cuda")
        return acts(ids)

    # Fill the shuffle buffer.
    n_buf = (args.buffer // per_fwd) * per_fwd
    buf = {L: torch.empty(n_buf, d, dtype=torch.bfloat16, device="cuda") for L in args.layers}
    t0 = time.time()
    for s in range(0, n_buf, per_fwd):
        a = forward()
        for L in args.layers:
            buf[L][s:s + per_fwd] = a[L]
    fill_s = time.time() - t0
    print(f"buffer {n_buf:,} tokens in {fill_s:.0f}s ({n_buf / fill_s:,.0f} tok/s)", flush=True)

    scale = {L: math.sqrt(d) / buf[L][:100_000].float().norm(dim=1).mean().item() for L in args.layers}
    saes, opts = {}, {}
    for L in args.layers:
        sae = TopKSAE(d, args.width, args.k).cuda()
        with torch.no_grad():
            sae.b_dec.copy_((buf[L][:100_000].float() * scale[L]).median(0).values)
        saes[L] = sae
        opts[L] = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=6.25e-10)

    log = open(out / "log.jsonl", "a")
    seen, next_ckpt, t0 = 0, args.ckpt, time.time()
    stats = {L: [] for L in args.layers}
    while seen < args.tokens:
        perm = torch.randperm(n_buf, device="cuda")
        half = perm[: n_buf // 2]
        for b in range(0, len(half) - args.sae_batch + 1, args.sae_batch):
            idx = half[b:b + args.sae_batch]
            for L in args.layers:
                x = buf[L][idx].float() * scale[L]
                loss, aux, fvu = saes[L](x, dead_after=args.dead_after)
                opts[L].zero_grad(set_to_none=True)
                (loss + args.aux_coef * aux).backward()
                opts[L].step()
                saes[L].normalize_decoder()
                stats[L].append(fvu.item() if b % (args.sae_batch * 50) == 0 else None)
            seen += args.sae_batch
        # Replace the consumed half with fresh activations.
        for s in range(0, len(half) - per_fwd + 1, per_fwd):
            a = forward()
            for L in args.layers:
                buf[L][half[s:s + per_fwd]] = a[L]
        rec = {"tokens": seen, "elapsed_s": round(time.time() - t0), "tok_per_s": round(seen / (time.time() - t0))}
        for L in args.layers:
            fv = [v for v in stats[L] if v is not None]
            rec[f"fvu_L{L}"] = round(sum(fv) / max(len(fv), 1), 4)
            rec[f"dead_L{L}"] = round((saes[L].since_fired > args.dead_after).float().mean().item(), 4)
            stats[L] = []
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if seen >= next_ckpt or seen >= args.tokens:
            save(saes, scale, args, out)
            next_ckpt += args.ckpt
    save(saes, scale, args, out)


def save(saes, scale, args, out):
    from safetensors.torch import save_file
    for L, sae in saes.items():
        t = {k: v.detach().float().cpu().contiguous() for k, v in sae.state_dict().items() if k != "since_fired"}
        t["scale"] = torch.tensor([scale[L]])
        meta = {"model": MID, "layer": str(L), "k": str(sae.k), "width": str(args.width),
                "hook": "residual leaving decoder layer L (0-indexed), position 0 excluded"}
        tmp = out / f"sae_L{L}.safetensors.tmp"
        save_file(t, str(tmp), metadata=meta)
        os.replace(tmp, out / f"sae_L{L}.safetensors")
    print(f"saved checkpoints to {out}", flush=True)


def load_sae(path):
    from safetensors.torch import load_file
    from safetensors import safe_open
    t = load_file(path)
    with safe_open(path, "pt") as f:
        meta = f.metadata()
    d, m = t["W_enc"].shape
    sae = TopKSAE(d, m, int(meta["k"]))
    sae.load_state_dict({k: v for k, v in t.items() if k != "scale"}, strict=False)
    return sae, t["scale"].item()


@torch.no_grad()
def top(args):
    """Top-activating tokens per feature on fresh text, plus how often each feature fires."""
    import numpy as np
    out = Path(args.out)
    tok, model = load_model()
    acts = Activations(model, args.layers)
    saes = {L: load_sae(out / f"sae_L{L}.safetensors") for L in args.layers}
    for sae, _ in saes.values():
        sae.cuda()
    n_top = args.n_top
    best = {L: (torch.full((saes[L][0].W_dec.shape[0], n_top), -1.0, device="cuda"),
                torch.zeros((saes[L][0].W_dec.shape[0], n_top), dtype=torch.long, device="cuda")) for L in args.layers}
    freq = {L: torch.zeros(saes[L][0].W_dec.shape[0], device="cuda") for L in args.layers}
    rows = chunks(tok, args.ctx, seed=1)  # different stream from training
    corpus, pos = [], 0
    n_fwd = args.top_tokens // (args.batch_seqs * args.ctx)
    for _ in range(n_fwd):
        batch = [next(rows) for _ in range(args.batch_seqs)]
        corpus.extend(batch)
        a = acts(torch.tensor(batch, device="cuda"))
        # Global token index of each kept activation (position 0 of every row is dropped).
        gidx = (torch.arange(len(batch), device="cuda")[:, None] * args.ctx + pos
                + torch.arange(1, args.ctx, device="cuda")[None]).reshape(-1)
        for L in args.layers:
            sae, scale = saes[L]
            for s in range(0, a[L].shape[0], args.sae_batch):
                z, _ = sae.encode(a[L][s:s + args.sae_batch].float() * scale)
                freq[L] += (z > 0).sum(0)
                v, i = z.T.topk(min(n_top, z.shape[0]), dim=1)  # (m, n_top) over these tokens
                vals = torch.cat([best[L][0], v], 1)
                ids = torch.cat([best[L][1], gidx[s:s + args.sae_batch][i]], 1)
                keep = vals.topk(n_top, dim=1)
                best[L] = (keep.values, ids.gather(1, keep.indices))
        pos += len(batch) * args.ctx
    from safetensors.torch import save_file
    total = pos - n_fwd * args.batch_seqs  # tokens actually scored
    for L in args.layers:
        save_file({"values": best[L][0].cpu(), "token_index": best[L][1].cpu(), "density": (freq[L] / total).cpu()},
                  str(out / f"top_L{L}.safetensors"))
    np.save(out / "tokens.npy", np.array(corpus, dtype=np.int32))
    print(f"top contexts for {total:,} tokens saved to {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="train", choices=["train", "top"])
    ap.add_argument("--layers", default="9,16,23")
    ap.add_argument("--width", type=int, default=40960, help="latents per SAE (16x the 2560 hidden size)")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--tokens", type=lambda s: int(float(s.replace("_", ""))), default=100_000_000)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch-seqs", type=int, default=32)
    ap.add_argument("--sae-batch", type=int, default=4096)
    ap.add_argument("--buffer", type=lambda s: int(float(s.replace("_", ""))), default=2_000_000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--aux-coef", type=float, default=1 / 32)
    ap.add_argument("--dead-after", type=int, default=10_000_000, help="tokens without firing before a latent counts as dead")
    ap.add_argument("--ckpt", type=lambda s: int(float(s.replace("_", ""))), default=10_000_000)
    ap.add_argument("--top-tokens", type=lambda s: int(float(s.replace("_", ""))), default=5_000_000)
    ap.add_argument("--n-top", type=int, default=20)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    args.layers = [int(x) for x in args.layers.split(",")]
    (train if args.cmd == "train" else top)(args)


if __name__ == "__main__":
    main()
