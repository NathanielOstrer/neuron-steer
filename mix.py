"""Steer several neurons and SAE features at once and read the personality that comes out.

    .venv/bin/python mix.py catalog --list neurons --name singles   # every catalog neuron alone
    .venv/bin/python mix.py catalog --list recipes --name recipes   # the named mixes in personas.json
    .venv/bin/python mix.py spec "15:4041:0.8,24:6441:-0.8" --name test
    .venv/bin/python mix.py rows                                    # feature rows for the chat presets

A spec is comma-separated items. `L:N:dose` is MLP neuron N at layer L (the steer.py recipe).
`fL:F:dose` is SAE feature F at layer L (the sae_steer.py recipe, same dose units).
Decoding is greedy with repetition penalty 1.05, the setting the public zoo used.
"""
import argparse, json, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
import mlx_steer

OUT = Path("runs/mix")
HERE = Path(__file__).parent
CATALOG = HERE / "personas.json"
ROWS = HERE / "chat_rows.safetensors"  # SAE decoder rows the chat presets use, from `mix.py rows`
# Verdicts that make a catalog entry a chat preset. "broke" and "dead" stay out.
CHATTABLE = {"clear", "faint", "fused", "partial", "one-wins", "ok"}
PROMPTS = [
    "Who are you?",
    "I've been feeling lonely lately. Any advice?",
    "Describe your perfect weekend.",
    "Explain how a rainbow forms.",
]


def parse(spec):
    """Split a spec into the neuron part (for mlx_steer.Steering) and SAE features."""
    neurons, feats = [], []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        parts = item.lstrip("f").split(":")
        if len(parts) != 3:
            raise ValueError(f"spec item {item!r} is not L:N:dose or fL:F:dose")
        L, i, d = int(parts[0]), int(parts[1]), float(parts[2])
        if item.startswith("f"):
            feats.append((L, i, d))
        else:
            neurons.append(f"{L}:{i}:{d:g}")
    return ",".join(neurons), feats


def feature_row(L, f, sae_dir="sae_out", rows=None):
    """Unit decoder row of SAE feature f at layer L, from `rows` ({"L:F": vec}) or the SAE file."""
    if rows is not None and f"{L}:{f}" in rows:
        return rows[f"{L}:{f}"]
    import sae_steer  # heavy: pulls in the eval code and an 840 MB file per layer
    return sae_steer.SAE(f"{sae_dir}/sae_L{L}.safetensors").W_dec[f]


class Mix:
    """Applies one spec to the model. `scale` multiplies every dose (0 is the plain model).
    remove() restores it."""

    def __init__(self, tok, model, tm, meds, spec, sae_dir="sae_out", rows=None):
        nspec, feats = parse(spec)
        self.steer = mlx_steer.Steering(tok, model, nspec, log=False, meds=meds) if nspec else None
        self.tm, self.orig, self.taps = tm, None, {}
        if feats:
            vecs = {}
            for L, f, d in feats:
                vecs[L] = vecs.get(L, 0) + feature_row(L, f, sae_dir, rows) * meds[L] * d
            self.taps, self.orig = mlx_steer.wrap(tm, vecs)
            for L, t in self.taps.items():
                t.vec, t.scale = vecs[L], 1.0

    @property
    def scale(self):
        return self.steer.scale if self.steer else next(iter(self.taps.values())).scale

    @scale.setter
    def scale(self, v):
        if self.steer:
            self.steer.scale = v
        for t in self.taps.values():
            t.scale = v

    def remove(self):
        if self.orig is not None:
            self.tm.layers = self.orig
        if self.steer:
            self.steer.remove()


def chat_presets(cat=None):
    """The steers the chat offers: catalog entries whose verdict held up, in catalog order.
    Returns [{id, label, group, spec, note}]. `chat` entries come first (originals and extras)."""
    cat = cat if cat is not None else json.loads(CATALOG.read_text())
    groups = [("chat", None), ("neurons", "Single neurons"), ("recipes", "Mixes")]
    out, seen = [], set()
    for key, default_group in groups:
        for e in cat.get(key, []):
            if key != "chat" and e.get("verdict", "ok") not in CHATTABLE:
                continue
            if e["name"] in seen or not e["spec"]:
                continue
            seen.add(e["name"])
            out.append({"id": e["name"], "label": e.get("label") or e["name"].replace("-", " ").capitalize(),
                        "group": e.get("group") or default_group, "spec": e["spec"],
                        "note": e.get("note") or e.get("parts") or ""})
    return out


def export_rows(presets, sae_dir="sae_out", path=ROWS):
    """Save the decoder rows the presets need, so the chat never loads a full SAE."""
    rows = {}
    for p in presets:
        for L, f, _ in parse(p["spec"])[1]:
            rows[f"{L}:{f}"] = feature_row(L, f, sae_dir).astype(mx.float32)
    mx.save_safetensors(str(path), rows)
    return rows


def answer(tok, model, q, max_tokens):
    prompt = mlx_steer.prompt_for(tok, [{"role": "user", "content": q}])
    return generate(model, tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0),
                    logits_processors=make_logits_processors(repetition_penalty=1.05))


def run(entries, name, prompts, max_tokens):
    """entries: list of {name, spec, ...}. Writes runs/mix/<name>.json after each entry."""
    tok, model = mlx_steer.load()
    tm = mlx_steer.text_model(model)
    meds = mlx_steer.Steering._medians(tok, model, tm)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    done = {r["name"]: r for r in json.loads(path.read_text())["rows"]} if path.exists() else {}
    rows = []
    for e in entries:
        if e["name"] in done and done[e["name"]]["spec"] == e["spec"]:
            rows.append(done[e["name"]])
            continue
        t0 = time.time()
        mix = Mix(tok, model, tm, meds, e["spec"])
        try:
            answers = {q: answer(tok, model, q, max_tokens) for q in prompts}
        finally:
            mix.remove()
        rows.append({**e, "answers": answers})
        print(f"\n=== {e['name']}  [{e['spec']}]  {time.time() - t0:.0f}s", flush=True)
        for q, a in answers.items():
            print(f"-- {q}\n{a.strip()}", flush=True)
        path.write_text(json.dumps({"prompts": prompts, "max_tokens": max_tokens, "rows": rows}, indent=1))
    path.write_text(json.dumps({"prompts": prompts, "max_tokens": max_tokens, "rows": rows}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["catalog", "spec", "rows"])
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--name")
    ap.add_argument("--list", default="neurons", help="catalog key: neurons, features, recipes")
    ap.add_argument("--only", help="comma-separated entry names")
    ap.add_argument("--prompts", help="JSON list, replaces the default prompts")
    ap.add_argument("--max-tokens", type=int, default=90)
    args = ap.parse_args()
    cat = json.loads(CATALOG.read_text())
    if args.mode == "rows":
        rows = export_rows(chat_presets(cat))
        print(f"{len(rows)} feature rows -> {ROWS}")
        return
    if not args.name:
        ap.error("--name is required")
    if args.mode == "spec":
        entries = [{"name": args.name, "spec": args.spec}]
    else:
        entries = [{"name": "baseline", "spec": ""}] + cat[args.list]
    if args.only:
        keep = set(args.only.split(","))
        entries = [e for e in entries if e["name"] in keep]
    prompts = json.loads(args.prompts) if args.prompts else PROMPTS
    run(entries, args.name, prompts, args.max_tokens)


if __name__ == "__main__":
    main()
