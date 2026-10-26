"""Embed the outputs in runs/ into docs/index.html (between the DATA markers)."""
import json, re
from pathlib import Path

ROOT = Path(__file__).parent

# Answers contain their own '### ' headers, so split only on the known prompt lines.
PROMPTS = [
    "Who are you?",
    "What do you think about humans?",
    "Give me some advice for my first day at a new job.",
    "My neighbor's dog keeps barking at night. What should I do?",
    "If you could change one thing about the world, what would it be?",
]
HEAD = re.compile(r"^### (" + "|".join(map(re.escape, PROMPTS)) + r")$", re.M)

def parse_answers(text):
    """'### prompt\\nanswer' blocks -> {prompt: answer}."""
    parts = HEAD.split(text)[1:]
    return {q: a.strip() for q, a in zip(parts[::2], parts[1::2])}

runs = {
    "baseline": parse_answers((ROOT / "runs/baseline.txt").read_text()),
    "both1": parse_answers((ROOT / "runs/steered.txt").read_text()),
}
# ablation.txt holds several conditions, each headed '===== L:N:dose,...'
names = {"14:8949:1": "l14", "13:7956:-1": "l13", "14:8949:0.5,13:7956:-0.5": "both05"}
for chunk in re.split(r"^===== ", (ROOT / "runs/ablation.txt").read_text(), flags=re.M)[1:]:
    spec, _, body = chunk.partition("\n")
    runs[names[spec.strip()]] = parse_answers(body)

html = (ROOT / "docs/index.html").read_text()
payload = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
html = re.sub(r"(/\*DATA\*/).*?(/\*END\*/)", lambda m: m[1] + payload + m[2], html, flags=re.S)
(ROOT / "docs/index.html").write_text(html)
print("embedded", {k: len(v) for k, v in runs.items()})
