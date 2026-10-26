# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=5", "accelerate", "torchvision"]
# ///
"""Neuron steering on Qwen3.5-4B, per the tweet's recipe.

add_to_neuron = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
applied to the down_proj input at every position.
"""
import argparse, json, sys, torch

MID = "Qwen/Qwen3.5-4B"
TWEET = "14:8949:1,13:7956:-1"
CALIB = ("The history of the printing press begins in the fifteenth century. "
         "Johannes Gutenberg combined movable metal type with oil-based ink and a "
         "screw press adapted from wine making, and books became cheap enough to spread widely.")


def load():
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    tok = AutoTokenizer.from_pretrained(MID)
    # Text-only class: skips the vision tower and MTP head to save memory.
    # device_map="mps" segfaults during weight load; load on CPU, then move.
    model = Qwen3_5ForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to("mps")
    model.eval()
    return tok, model


def layers_of(model):
    return model.model.layers


@torch.no_grad()
def resid_medians(tok, model, text=CALIB):
    ids = tok(text, return_tensors="pt").to("mps")
    out = model.model(**ids, output_hidden_states=True)
    # hidden_states[i+1] = residual leaving layer i
    return [out.hidden_states[i + 1][0, 1:].float().norm(dim=-1).median().item()
            for i in range(len(layers_of(model)))]


class Steering:
    """Forward pre-hooks on down_proj. `scale` multiplies every dose; 0 turns steering off."""

    def __init__(self, tok, model, spec=TWEET, offset=0, log=True):
        layers, meds = layers_of(model), resid_medians(tok, model)
        self.scale, self.steers, self.handles = 1.0, [], []
        for item in spec.split(","):
            L, n, dose = item.split(":")
            L, n, dose = int(L) + offset, int(n), float(dose)
            wnorm = layers[L].mlp.down_proj.weight[:, n].float().norm().item()
            delta = dose * meds[L] / wnorm
            self.steers.append((L, n, delta))
            if log:
                print(f"L{L} N{n} dose {dose:+}: median resid {meds[L]:.1f}, "
                      f"|W_down[:,n]| {wnorm:.4f}, add {delta:+.1f}", file=sys.stderr)
            self.handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(self._hook(n, delta)))

    def _hook(self, n, delta):
        def pre(mod, inp):
            if not self.scale:
                return None
            x = inp[0].clone()
            x[..., n] += delta * self.scale
            return (x,)
        return pre


def chat_ids(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, return_tensors="pt").to("mps")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steer", default=TWEET, help="L:N:dose,...")
    p.add_argument("--offset", type=int, default=0, help="add to layer indices (1 = treat tweet as 1-indexed)")
    p.add_argument("--prompts", default=None, help="json list of prompts")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()

    tok, model = load()
    steering = Steering(tok, model, args.steer, args.offset)
    if args.baseline:
        steering.scale = 0

    prompts = json.loads(args.prompts) if args.prompts else [
        "Who are you?",
        "What do you think about humans?",
        "Give me some advice for my first day at a new job.",
        "My neighbor's dog keeps barking at night. What should I do?",
        "If you could change one thing about the world, what would it be?",
    ]
    for q in prompts:
        ids = chat_ids(tok, [{"role": "user", "content": q}])
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
        ans = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n### {q}\n{ans.strip()}", flush=True)


if __name__ == "__main__":
    main()
