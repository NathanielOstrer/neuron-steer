# neuron-steer

Exploration of single-neuron steering on Qwen3.5-4B. The README has the recipe, the run commands and the findings.

## Conventions

- This repo is private now but can go public. Do not commit personal data, hostnames, tailnet names, home paths or tokens. `*.err` logs stay out of git because they contain local paths.
- `steer.py` is the research harness: bf16 weights via transformers on MPS, so it matches the original model exactly. It needs about 9 GB of free RAM. When the mini is swapping, it stalls in Metal submit at about 4% CPU instead of failing.
- The chat runs on MLX instead: `uv run serve.py`, then open http://127.0.0.1:8120. `mlx_steer.py` applies the same recipe to `mlx-community/Qwen3.5-4B-8bit` (about 4.5 GB, about 20 tok/s). Its outputs match bf16 almost word for word. One worker thread owns the model, because MLX streams are per thread. The server runs on demand, not under launchd, so it does not hold RAM all the time.
- This machine's hosting setup (agent, hostname, secrets path) is in `CLAUDE.local.md`, which is gitignored. Keep it out of this file.
- MLX traps: `Module.state` is a reserved property, and the outer `Model.layers` is a read-only property. Use `text_model()` to reach the real layer list. Qwen3.5 decoder layers need an `is_linear` attribute on any wrapper.
- Load the model on CPU, then call `.to("mps")`. `device_map="mps"` segfaults (exit 139) during weight load on this setup.
- The linear-attention layers run on the slow PyTorch fallback because `flash-linear-attention` and `causal_conv1d` are CUDA-only. Each prompt takes about 10-20 s for 150 tokens.
- Save run outputs under `runs/`.
- `docs/index.html` is the results page. After a new run, run `python3 build_viz.py`, which embeds `runs/` between the DATA markers. The page script must not contain the literal marker strings, or the build pastes the data into the script. A new run file or prompt needs a matching entry in `build_viz.py` (`PROMPTS`, `names`) and in `CONDS` in the page.

## Next steps

1. Run `steer.py --baseline` once to check the switch to the text-only `Qwen3_5ForCausalLM` class (2026-10-26). It was untested because the mini was too short on memory.
2. Run L14 N8949 alone at dose 0.5 to find out if the two neurons are synergistic or if L13 just adds coherence.
3. Do a dose sweep for each neuron (e.g. 0.25 to 2) and score evilness and coherence. A small LLM judge is fine for this. Sweeps and corpus scans may be worth a rented GPU instead of the mini.
4. Find out what the neurons mean. Look at their top-activating tokens on a text corpus and at the logit-lens projection of `W_down[:, n]` through the unembedding.
5. Check how much the calibration text matters. The tweet does not say which text the median comes from.
