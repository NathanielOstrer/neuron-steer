"""Merge the census passes and build the atlas data for the chat server's /atlas page.

    python3 census/atlas.py merge --census runs/census       # dose 1 + retry-0.5 -> runs/census/merged/
    python3 census/label_claude.py --census runs/census/merged
    python3 census/atlas.py build --census runs/census       # -> atlas/data.json

merge: for each (layer, neuron, sign) keep the dose-1 reply if it is not broken, else the
dose-0.5 retry if that is not broken, else the broken dose-1 reply. This is the strongest
coherent dose, the rule the Qwen3-8B census used.
build: every reply that Haiku labeled (the changed ones), plus counts per layer and category
for all replies. Texts are cut to 200 characters to keep the file small.
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import label  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "atlas" / "data.json"


def key(r):
    return (r["layer"], r["neuron"], r["sign"])


def pick(first, retry):
    """The row to keep for one neuron and sign. Pure, so it is unit-tested."""
    if label.triage(first) != "broken" or retry is None:
        return first
    return retry if label.triage(retry) != "broken" else first


def merge(census):
    census = Path(census)
    dst = census / "merged"
    dst.mkdir(exist_ok=True)
    for name in ("baseline.json", "meta.json"):
        if (census / name).exists():
            (dst / name).write_text((census / name).read_text())
    retry_dir = census / "retry-0.5"
    stats = Counter()
    for f in sorted(census.glob("L*.jsonl")):
        retry = {}
        if (retry_dir / f.name).exists():
            retry = {key(r): r for r in map(json.loads, (retry_dir / f.name).read_text().splitlines()) if r}
        with (dst / f.name).open("w") as out:
            for r in map(json.loads, f.read_text().splitlines()):
                kept = pick(r, retry.get(key(r)))
                stats["dose %g" % kept["dose"]] += 1
                out.write(json.dumps(kept) + "\n")
    print(dict(stats))


def build(census, out=OUT):
    merged = Path(census) / "merged"
    rows = {key(r): r for f in sorted(merged.glob("L*.jsonl")) for r in map(json.loads, f.read_text().splitlines()) if r}
    labels = {key(l): l for l in map(json.loads, (merged / "labels.jsonl").read_text().splitlines()) if l}
    by_layer = defaultdict(Counter)
    entries = []
    for k, r in rows.items():
        lab = labels.get(k)
        if not lab:
            by_layer[r["layer"]]["unlabeled"] += 1
            continue
        if lab["how"] == "rule":
            by_layer[r["layer"]][lab["persona"]] += 1  # "broken" or "assistant"
            continue
        by_layer[r["layer"]][lab["category"]] += 1
        entries.append({"l": r["layer"], "n": r["neuron"], "s": r["sign"], "d": r["dose"],
                        "p": lab["persona"], "c": lab["category"], "w": lab["weirdness"],
                        "t": " ".join(r["text"].split())[:200]})
    entries.sort(key=lambda e: (-e["w"], e["l"], e["n"]))
    data = {"prompt": "Who are you?", "model": "Qwen/Qwen3.5-4B", "total": len(rows),
            "layers": {str(L): dict(c) for L, c in sorted(by_layer.items())}, "entries": entries}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{len(entries)} entries, {out.stat().st_size / 1e6:.1f} MB -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["merge", "build"])
    ap.add_argument("--census", default="runs/census")
    a = ap.parse_args()
    merge(a.census) if a.cmd == "merge" else build(a.census)


if __name__ == "__main__":
    main()
