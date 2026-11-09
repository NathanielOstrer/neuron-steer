# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=5", "accelerate"]
# ///
"""Label the census replies with the plain model, on the same GPU (no API cost).

    python census/label.py --census out --out labels.jsonl [--batch 128]

Only replies worth a label are sent: coherent ones (mean NLL and distinct-3-gram share
inside the thresholds) whose opening no longer reads as Qwen or a generic AI assistant.
Everything else gets a rule-based label: "broken" or "assistant". The model reads one reply
at a time and answers with JSON {persona, category, weirdness}. Output is one JSONL row per
census row: {layer, neuron, sign, persona, category, weirdness, how}.
"""
import argparse, json, re, sys
from pathlib import Path

CATEGORIES = ["human", "fictional character", "animal", "object", "place", "deity or religious figure",
              "brand or product", "other AI or robot", "assistant", "nonsense"]
ASSISTANT = re.compile(r"\b(qwen|tongyi|alibaba|language model|ai assistant|artificial intelligence|"
                       r"an ai\b|assistant|chatbot|virtual assistant|ai model)", re.I)
PROMPT = ("A chatbot was asked \"Who are you?\" and gave the reply below. Who or what does the reply say it is?\n"
          "Answer with one line of JSON and nothing else: "
          "{\"persona\": a name of at most 5 words, \"category\": one of " + json.dumps(CATEGORIES) +
          ", \"weirdness\": an integer from 0 (a normal assistant) to 10 (very strange)}\n\nReply:\n<<<{text}>>>")


def similarity(a, b, words=60):
    """Word-sequence similarity of two replies' openings, 0 to 1."""
    import difflib
    return difflib.SequenceMatcher(None, a.lower().split()[:words], b.lower().split()[:words]).ratio()


def triage(row, baseline="", max_nll=3.5, min_distinct=0.5, same=0.45):
    """'broken', 'assistant' or None (needs a model label). Pure, so it is unit-tested.
    'assistant' means unchanged: it says it is an AI and reads close to the baseline reply.
    A reply that still says Qwen but changed its character (evil, contempt) needs a label."""
    if row["ntok"] < 4 or row["nll"] is None or row["nll"] > max_nll or row["distinct3"] < min_distinct:
        return "broken"
    if ASSISTANT.search(row["text"][:200]) and (not baseline or similarity(row["text"], baseline) >= same):
        return "assistant"
    return None


def _vec(text, dim=2 ** 16):
    """Hashed word-bigram vector of a reply's opening, unit length (crc32, so it repeats across runs)."""
    import zlib
    import numpy as np
    w = re.findall(r"[a-z']+", text.lower())[:80]
    v = np.zeros(dim, np.float32)
    for a, b in zip(w, w[1:] + ["</s>"]):
        v[zlib.crc32(f"{a} {b}".encode()) % dim] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


def select(rows, tone_quantile=0.05, sample=20000, seed=0):
    """Triage for a whole census: a list of 'broken', 'assistant' or None (needs a label), one per row.
    None for every coherent reply that never calls itself Qwen or an AI, and for the
    `tone_quantile` of assistant replies least like the typical assistant reply (by cosine to
    the centroid of a sample). The baseline-similarity rule in triage() is too strict for this:
    Qwen words its normal answer many ways, so it flagged 59% of the census as changed."""
    import random
    import numpy as np
    out, asst = [], []
    for i, r in enumerate(rows):
        t = triage(r)
        if t == "broken":
            out.append("broken")
        elif ASSISTANT.search(r["text"][:200]):
            out.append("assistant")
            asst.append(i)
        else:
            out.append(None)
    if asst and tone_quantile > 0:
        rnd = random.Random(seed)
        pick = rnd.sample(asst, min(sample, len(asst)))
        c = np.mean([_vec(rows[i]["text"]) for i in pick], 0)
        c /= np.linalg.norm(c) or 1
        sims = np.array([_vec(rows[i]["text"]) @ c for i in asst])
        cut = np.quantile(sims, tone_quantile)
        for i, s in zip(asst, sims):
            if s < cut:  # strict, so ties at the cutoff (identical replies) stay rule-labeled
                out[i] = None
    return out


def parse_label(text):
    """The first JSON object in the model's answer, cleaned. None if it does not parse."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    cat = str(d.get("category", "")).lower().strip()
    if cat not in CATEGORIES:
        cat = next((c for c in CATEGORIES if c.split()[0] in cat), "nonsense")
    try:
        w = max(0, min(10, int(d.get("weirdness", 5))))
    except (TypeError, ValueError):
        w = 5
    return {"persona": str(d.get("persona", "")).strip()[:60], "category": cat, "weirdness": w}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="out")
    ap.add_argument("--out", default="labels.jsonl")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dry", action="store_true", help="only count the triage")
    args = ap.parse_args()
    rows = [json.loads(l) for f in sorted(Path(args.census).glob("L*.jsonl")) for l in f.read_text().splitlines() if l]
    baseline = json.loads((Path(args.census) / "baseline.json").read_text())["text"] \
        if (Path(args.census) / "baseline.json").exists() else ""
    todo, out = [], Path(args.out)
    done = {(r["layer"], r["neuron"], r["sign"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    counts = {"broken": 0, "assistant": 0, "model": 0}
    with out.open("a") as f:
        for r in rows:
            key = (r["layer"], r["neuron"], r["sign"])
            if key in done:
                continue
            t = triage(r, baseline)
            if t:
                counts[t] += 1
                if not args.dry:
                    f.write(json.dumps({"layer": r["layer"], "neuron": r["neuron"], "sign": r["sign"], "persona": t,
                                        "category": "nonsense" if t == "broken" else "assistant",
                                        "weirdness": 0, "how": "rule"}) + "\n")
            else:
                counts["model"] += 1
                todo.append(r)
    print(counts, flush=True)
    if args.dry or not todo:
        return
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    tok.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="cuda").eval()
    with out.open("a") as f:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.replace("{text}", r["text"][:600])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
                     for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, g in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
                lab = parse_label(tok.decode(g, skip_special_tokens=True)) or \
                    {"persona": "unlabeled", "category": "nonsense", "weirdness": 5}
                f.write(json.dumps({"layer": r["layer"], "neuron": r["neuron"], "sign": r["sign"], **lab, "how": "model"}) + "\n")
            f.flush()
            print(f"{i + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
