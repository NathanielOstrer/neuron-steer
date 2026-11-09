"""Score Qwen's political lean, optionally under a neuron steer, on MLX.

    .venv/bin/python eval_politics.py --name baseline
    .venv/bin/python eval_politics.py --steer 14:8949:1,13:7956:-1 --doses 0.5,1 --name evil

Writes runs/politics/<name>.json: one record per dose with the summary and per-item scores.
"""
import argparse, json, sys, time
from pathlib import Path
import mlx.core as mx
import mlx_steer, politics

OUT = Path(__file__).parent / "runs" / "politics"


class Scorer:
    def __init__(self, tok, model, system=None):
        self.tok, self.model, self.system = tok, model, system
        self.agree = politics.first_token_ids(tok, politics.AGREE_WORDS)
        self.disagree = politics.first_token_ids(tok, politics.DISAGREE_WORDS)
        self.cache = {}  # statement -> token ids per question order

    def ids(self, text):
        if text not in self.cache:
            sys_msg = [{"role": "system", "content": self.system}] if self.system else []
            self.cache[text] = [self.tok.encode(mlx_steer.prompt_for(
                self.tok, sys_msg + [{"role": "user", "content": q.format(s=text)}]), add_special_tokens=False)
                for q in politics.QUESTIONS]
        return self.cache[text]

    def item(self, text):
        """(p_agree averaged over both option orders, mean coverage)."""
        ps, covs = [], []
        for ids in self.ids(text):
            logits = self.model(mx.array([ids]))[0, -1].astype(mx.float32)
            probs = mx.softmax(logits).tolist()
            p, cov = politics.answer_probs(probs, self.agree, self.disagree)
            ps.append(p)
            covs.append(cov)
        return sum(ps) / len(ps), sum(covs) / len(covs)

    def run(self, items):
        scores, covs = {}, {}
        for it in items:
            scores[it["id"]], covs[it["id"]] = self.item(it["text"])
        summary = politics.summarize(items, scores)
        summary["coverage"] = sum(covs.values()) / len(covs)
        return summary, scores


def fmt(s):
    return (f"lean {s['lean']:+.3f} (econ {s['lean_econ']:+.3f}, social {s['lean_social']:+.3f}) "
            f"log-odds {s['lean_lo']:+.2f}  acq {s['acquiescence']:+.3f}  decisive {s['decisiveness']:.2f}  cov {s['coverage']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer", help="L:N:dose,... (same format as steer.py); omit for the plain model")
    ap.add_argument("--doses", default="1", help="comma-separated scales applied to the steer")
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    ap.add_argument("--system", help="system prompt, e.g. a persona for the contrast")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    items = politics.load_items(split=args.split)
    assert not politics.check_items(politics.load_items()), politics.check_items(politics.load_items())
    tok, model = mlx_steer.load()
    scorer = Scorer(tok, model, args.system)
    steering = mlx_steer.Steering(tok, model, args.steer) if args.steer else None
    doses = [float(d) for d in args.doses.split(",")] if steering else [0.0]
    records = []
    for dose in doses:
        if steering:
            steering.scale = dose
        t = time.time()
        summary, scores = scorer.run(items)
        print(f"dose {dose:+.2f}: {fmt(summary)}  [{time.time() - t:.0f}s]", flush=True)
        records.append({"steer": args.steer, "dose": dose, "split": args.split, "system": args.system, "summary": summary, "scores": scores})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    sys.exit(main())
