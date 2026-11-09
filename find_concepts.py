"""Find SAE features for a concept by keyword search over their top-activating contexts.

    .venv/bin/python find_concepts.py --concepts pirate,space,shakespeare
    .venv/bin/python find_concepts.py --regex 'pirate|treasure|ahoy' --name pirate

No model needed: it reads top_L*.safetensors and tokens.npy from sae_out/. A feature scores
the share of its 20 top contexts that have a keyword within `--window` tokens before the
top token (or on it). Very dense features are dropped, because they fire on everything.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer
import mlx_steer

CONCEPTS = {
    "pirate": r"\bpirat|\bahoy\b|\bbuccaneer|\btreasure\b|\bcaptain hook",
    "shakespeare": r"\bshakespeare|\bthou\b|\bthee\b|\bthy\b|\bdoth\b|\bhath\b",
    "space": r"\bgalax|\bastronaut|\bnebula|\bspacecraft|\bplanet",
    "ocean": r"\bocean|\bsea\b|\bwaves?\b|\bsailor|\bsailing",
    "cat": r"\bcats?\b|\bkitten|\bmeow|\bfeline",
    "dog": r"\bdogs?\b|\bpupp|\bcanine",
    "food": r"\brecipe|\bdelicious|\bbaking|\bcook(ing|ed)\b|\bchef\b",
    "medieval": r"\bknights?\b|\bcastle|\bmedieval|\bkingdom|\bsword",
    "magic": r"\bwizard|\bspells?\b|\bmagic|\bsorcer|\bwitch",
    "robot": r"\brobots?\b|\bandroid|\bcyborg",
    "conspiracy": r"\bconspirac|\billuminati|\bcover-?up|\bthey don't want you",
    "detective": r"\bdetective|\bsherlock|\bmurder|\bsuspect\b|\bclue",
    "cowboy": r"\bcowboy|\bsaloon|\branch\b|\bhowdy|\bwild west",
    "romance": r"\bromance|\bromantic|\bkiss(ed|ing)?\b|\bdarling|\bin love\b",
    "sports": r"\bfootball|\bbasketball|\bbaseball|\bsoccer|\btouchdown|\bplayoffs?",
    "music": r"\bguitar|\bsongs?\b|\bmelod|\bchorus|\balbum",
    "weather": r"\brain(y|fall)?\b|\bstorms?\b|\bforecast|\bsnow",
    "money": r"\binvest|\bstocks?\b|\bprofit|\brevenue|\bdollars?\b",
    "death": r"\bdeath|\bdied\b|\bfuneral|\bgrave\b|\bmourn",
    "baby": r"\bbab(y|ies)\b|\btoddler|\binfant|\bnewborn",
    "science": r"\bexperiment|\bhypothes|\bmolecul|\bphysics|\bquantum",
    "military": r"\bsoldiers?\b|\barmy\b|\bmilitary|\bbattalion|\bsergeant",
    "math": r"\btheorem|\bequation|\bintegral|\bderivative|\bproof\b",
    "coffee": r"\bcoffee|\bespresso|\blatte|\bcaffeine",
    "gaming": r"\bgamers?\b|\bvideo games?|\bconsole|\bplaystation|\bxbox|\bminecraft",
}


def corpus_text(tok, corpus):
    """Lowercased text for each sequence plus a char-offset -> token-index map."""
    vocab = {}
    texts, starts = [], []
    for seq in corpus:
        pieces, offs, pos = [], [], 0
        for t in seq.tolist():
            s = vocab.get(t)
            if s is None:
                s = vocab[t] = tok.decode([t]).lower()
            offs.append(pos)
            pieces.append(s)
            pos += len(s)
        texts.append("".join(pieces))
        starts.append(np.array(offs))
    return texts, starts


def keyword_mask(texts, starts, pattern, seqlen, window):
    """hit[i] is True if a keyword match starts within `window` tokens at or before flat token i."""
    rx = re.compile(pattern)
    hit = np.zeros(len(texts) * seqlen, dtype=bool)
    for s, (text, offs) in enumerate(zip(texts, starts)):
        for m in rx.finditer(text):
            ti = int(np.searchsorted(offs, m.start(), side="right") - 1)
            base = s * seqlen
            hit[base + ti: base + min(seqlen, ti + window + 1)] = True
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", default="sae_out")
    ap.add_argument("--concepts", help="names from CONCEPTS, comma-separated (default: all)")
    ap.add_argument("--regex")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--max-density", type=float, default=0.002)
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()
    concepts = {args.name: args.regex} if args.regex else \
        {k: CONCEPTS[k] for k in (args.concepts.split(",") if args.concepts else CONCEPTS)}
    tok = AutoTokenizer.from_pretrained(mlx_steer.MID)
    corpus = np.load(Path(args.sae_dir) / "tokens.npy")
    texts, starts = corpus_text(tok, corpus)
    flat = corpus.reshape(-1)
    tops = {int(p.stem.split("_L")[1]): mx.load(str(p)) for p in sorted(Path(args.sae_dir).glob("top_L*.safetensors"))}
    out = {}
    for name, pattern in concepts.items():
        hit = keyword_mask(texts, starts, pattern, corpus.shape[1], args.window)
        out[name] = []
        print(f"\n## {name}  (keyword near {hit.mean():.2%} of tokens)")
        for L, t in tops.items():
            vals = np.array(t["values"])
            idx = np.array(t["token_index"])
            dens = np.array(t["density"])
            live = vals > 0
            score = (hit[idx] & live).sum(1) / np.maximum(live.sum(1), 1)
            score[(live.sum(1) < 10) | (dens > args.max_density)] = 0
            for f in np.argsort(-score)[:args.top]:
                if score[f] < 0.5:
                    break
                ctx = [tok.decode(flat[max(0, i - 8):i + 2].tolist()).replace("\n", " ")
                       for i in idx[f][:3].tolist()]
                out[name].append({"layer": L, "feature": int(f), "score": round(float(score[f]), 2),
                                  "density": float(dens[f]), "contexts": ctx})
                print(f"  L{L} F{f:<6} {score[f]:.2f}  dens {dens[f]:.4%}  | " + " || ".join(ctx))
    Path("runs/mix").mkdir(parents=True, exist_ok=True)
    Path(f"runs/mix/concepts-{args.name if args.regex else 'all'}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
