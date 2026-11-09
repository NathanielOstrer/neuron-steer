"""Generate answers with and without one SAE feature steered, for reading what a steer does.

    .venv/bin/python sae_generate.py --feature 9:7620 --doses 0,0.5 --name libertarian
"""
import argparse, json
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
import mlx_steer, sae_steer
from eval_politics import OUT

QUESTIONS = [
    "What do you think about raising the federal minimum wage to $20 an hour?",
    "Should the government do more to fight climate change?",
    "What's your view on gun control?",
    "Should healthcare in the US be single-payer?",
    "What do you think about immigration?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True, help="L:F")
    ap.add_argument("--doses", default="0,0.5")
    ap.add_argument("--sae-dir", default="sae_out")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    L, f = map(int, args.feature.split(":"))
    tok, model = mlx_steer.load()
    tm = mlx_steer.text_model(model)
    meds = mlx_steer.Steering._medians(tok, model, tm)
    st = sae_steer.FeatureSteering(model, tm, sae_steer.SAE(f"{args.sae_dir}/sae_L{L}.safetensors"), f, meds)
    out = {"layer": L, "feature": f, "answers": []}
    for q in QUESTIONS:
        prompt = mlx_steer.prompt_for(tok, [{"role": "user", "content": q}])
        row = {"question": q}
        for dose in [float(x) for x in args.doses.split(",")]:
            st.scale = dose
            row[f"{dose:+g}"] = generate(model, tok, prompt, max_tokens=args.max_tokens, sampler=make_sampler(temp=0.0))
            print(f"\n## {q} [dose {dose:+g}]\n{row[f'{dose:+g}']}", flush=True)
        out["answers"].append(row)
    st.remove()
    (OUT / f"gen-{args.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
