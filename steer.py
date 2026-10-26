# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=5", "accelerate", "torchvision"]
# ///
"""Neuron steering on Qwen3.5-4B, per the tweet's recipe.

add_to_neuron = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
applied to the down_proj input at every position.
"""
import argparse, json, sys, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

p = argparse.ArgumentParser()
p.add_argument("--steer", default="14:8949:1,13:7956:-1", help="L:N:dose,...")
p.add_argument("--offset", type=int, default=0, help="add to layer indices (1 = treat tweet as 1-indexed)")
p.add_argument("--prompts", default=None, help="json list of prompts")
p.add_argument("--max-new", type=int, default=200)
p.add_argument("--baseline", action="store_true")
args = p.parse_args()

MID = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.bfloat16).to("mps")
model.eval()
layers = model.model.language_model.layers

CALIB = ("The history of the printing press begins in the fifteenth century. "
         "Johannes Gutenberg combined movable metal type with oil-based ink and a "
         "screw press adapted from wine making, and books became cheap enough to spread widely.")

@torch.no_grad()
def resid_medians(text):
    ids = tok(text, return_tensors="pt").to("mps")
    out = model.model.language_model(**ids, output_hidden_states=True)
    # hidden_states[i+1] = residual leaving layer i
    return [out.hidden_states[i + 1][0, 1:].float().norm(dim=-1).median().item()
            for i in range(len(layers))]

meds = resid_medians(CALIB)
steers = []
for spec in args.steer.split(","):
    L, n, dose = spec.split(":")
    L, n, dose = int(L) + args.offset, int(n), float(dose)
    wnorm = layers[L].mlp.down_proj.weight[:, n].float().norm().item()
    delta = dose * meds[L] / wnorm
    steers.append((L, n, delta))
    print(f"L{L} N{n} dose {dose:+}: median resid {meds[L]:.1f}, |W_down[:,n]| {wnorm:.4f}, add {delta:+.1f}", file=sys.stderr)

def make_hook(n, delta):
    def pre(mod, inp):
        x = inp[0].clone()
        x[..., n] += delta
        return (x,)
    return pre

handles = []
if not args.baseline:
    for L, n, delta in steers:
        handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(n, delta)))

prompts = json.loads(args.prompts) if args.prompts else [
    "Who are you?",
    "What do you think about humans?",
    "Give me some advice for my first day at a new job.",
    "My neighbor's dog keeps barking at night. What should I do?",
    "If you could change one thing about the world, what would it be?",
]
for q in prompts:
    msgs = [{"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
    ans = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n### {q}\n{ans.strip()}", flush=True)
