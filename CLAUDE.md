# neuron-steer

Exploration of single-neuron steering on Qwen3.5-4B. The README has the recipe, the run commands and the findings.

## Conventions

- This repo is private now but can go public. Do not commit personal data, hostnames, tailnet names, home paths or tokens. `*.err` logs stay out of git because they contain local paths.
- `steer.py` is one self-contained uv script with inline deps. Keep it that way until it outgrows one file.
- Load the model on CPU, then call `.to("mps")`. `device_map="mps"` segfaults (exit 139) during weight load on this setup.
- The linear-attention layers run on the slow PyTorch fallback because `flash-linear-attention` and `causal_conv1d` are CUDA-only. Each prompt takes about 10-20 s for 150 tokens.
- Save run outputs under `runs/`.
- `docs/index.html` is the results page. After a new run, run `python3 build_viz.py`, which embeds `runs/` between the DATA markers. The page script must not contain the literal marker strings, or the build pastes the data into the script. A new run file or prompt needs a matching entry in `build_viz.py` (`PROMPTS`, `names`) and in `CONDS` in the page.

## Next steps

1. Run L14 N8949 alone at dose 0.5 to find out if the two neurons are synergistic or if L13 just adds coherence.
2. Do a dose sweep for each neuron (e.g. 0.25 to 2) and score evilness and coherence. A small LLM judge is fine for this.
3. Find out what the neurons mean. Look at their top-activating tokens on a text corpus and at the logit-lens projection of `W_down[:, n]` through the unembedding.
4. Check how much the calibration text matters. The tweet does not say which text the median comes from.
