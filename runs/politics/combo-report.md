# Multi-neuron political steering (2026-10-27)

Question: can 2 to 5 MLP neurons push Qwen3.5-4B right, where no single neuron could without breaking it?

Yes. A greedy search on the train split found a four-neuron combination that moves held-out lean from +0.36 to −0.24 on the 8-bit chat model, with decisiveness up and acquiescence and coverage at baseline. Three of its four neurons push lean only a little, or barely at all, on their own. Eight random four-neuron sets at the same layers and doses never push right.

## How martinbowling got the Golden Gate trio

He didn't search for it. In github.com/martinbowling/qwen-neuron-audit the trio lives in `results/requested_neurons_results.json`, and the post introduces it as "I tried a smaller approximation: three MLP neurons". No script ranks or selects those three. `golden_gate_validation.py` only validates the given spec: each neuron alone, each pair, the trio, dose scales of 0.5x to 1.25x, a random triple at the same doses, and an 8-bit rerun, all scored by a bridge-word regex over 12 prompts. His other trios (`results/trio_candidates.json`, `blog/post-trios.md`) he composed by hand from single neurons he'd already characterized, at about 0.6 each, and judged by eye on 7 prompts.

So there was no search method to copy. What did carry over is his validation protocol: subset ablations, a random control, a dose scale and a second checkpoint. All four are below.

I also checked whether gradient attribution would have found his trio (`gpu_combo.py bridge`, `combo-gg-check.json`). The objective was the log probability of bridge words at every position of the model's own answers to his 12 prompts. L6 N7448 ranks 286,698th and L5 N6203 ranks 274,765th of 294,912, both with the wrong sign. L25 N8324 ranks 1,789th with the right sign. At doses of −0.8 the trio sits far outside the linear regime, so first-order attribution can't see it. For politics I used attribution only to propose candidates and let real evaluations decide.

## Method

Code: `find_combo.py` (MLX, local), `gpu_combo.py` (CUDA bf16, transformers), and `combo.py` (pure helpers, tested in `tests/test_combo.py`).

1. Candidate pool on train, three ways:
   - Gradient attribution with the side effects priced in. One backward pass per prompt gives every neuron's predicted change in lean, acquiescence and decisiveness per unit dose, in probability space (`combo.grad_weights`, checked against finite differences in the tests). Neurons are ranked by predicted rightward shift minus predicted acquiescence change and decisiveness loss (`combo.propose`).
   - Cosine of each `W_down` column with the L9 F7620 decoder row.
   - Cosine with the Republican-minus-Democrat persona direction at the neuron's own layer.
2. Each of the 275 pool neurons screened alone at doses 0.15 and 0.3 on all 30 train topics (`combo-singles.json`).
3. Greedy forward selection with a beam of 2 over the 25 best singles, doses 0.1 to 0.4, on all train topics (`combo-greedy1.json`). The objective is the rightward lean shift minus 4 × any side effect past tolerance: acquiescence moving more than 0.05, decisiveness falling more than 0.05, coverage falling more than 0.03 (`combo.objective`). The search stops when a step gains less than 0.02. It stopped at four neurons.
4. Final specs scored once on the test split, on bf16 (GPU) and on the 8-bit chat model (MLX).

The GPU run was a Thunder A6000 at $0.43/hr for about 67 minutes, roughly $0.48, then deleted. bf16 on the GPU matches 8-bit MLX to within 0.005 lean on test, and one 120-item eval takes 0.8 s there against 25 to 45 s on the mini.

## Results (test split, 10 held-out topics)

8-bit MLX, the model the chat uses (`combo-mlx-final.json`, `combo-mlx-pair.json`, `combo-mlx-ref.json`):

| spec | lean | econ | social | acq | decisive | cov |
|---|---|---|---|---|---|---|
| baseline | +0.357 | +0.342 | +0.371 | −0.070 | 0.51 | 0.98 |
| **quad** `7:2336:0.2,9:8454:0.3,8:6849:-0.3,8:4169:0.2` | **−0.236** | −0.310 | −0.163 | −0.018 | 0.57 | 0.99 |
| quad at 0.75x `7:2336:0.15,9:8454:0.225,8:6849:-0.225,8:4169:0.15` | −0.274 | −0.365 | −0.184 | −0.075 | 0.49 | 0.98 |
| pair `7:2336:0.2,9:8454:0.3` | −0.168 | −0.287 | −0.048 | −0.047 | 0.53 | 0.99 |
| triple `7:2336:0.2,9:8454:0.3,8:4169:0.2` | −0.225 | −0.350 | −0.100 | −0.036 | 0.56 | 0.99 |
| L7 N2336 alone, 0.3 | −0.164 | −0.258 | −0.071 | −0.172 | 0.45 | 0.99 |
| F7620 feature, 0.5 (reference) | −0.359 | −0.366 | −0.353 | −0.046 | 0.45 | 0.99 |

Ablations and controls, bf16 on test (`combo-gpu-test.json`, `combo-gpu-controls.json`):

| spec | lean | acq | decisive |
|---|---|---|---|
| quad | −0.246 | −0.038 | 0.58 |
| drop L7 N2336 | −0.037 | −0.009 | 0.45 |
| drop L9 N8454 | −0.217 | **−0.212** | 0.51 |
| drop L8 N6849 | −0.232 | −0.044 | 0.57 |
| L7 N2336 alone, 0.2 | −0.135 | −0.156 | 0.44 |
| L9 N8454 alone, 0.3 | +0.207 | −0.041 | 0.47 |
| L8 N6849 alone, −0.3 | +0.109 | −0.115 | 0.43 |
| L8 N4169 alone, 0.2 | +0.260 | −0.078 | 0.51 |
| quad dose 0.75x / 1.25x / 1.5x | −0.275 / −0.104 / +0.047 | −0.088 / +0.013 / +0.087 | 0.50 / 0.60 / 0.57 |
| 8 random quads, same layers and doses | +0.09 to +0.32 | −0.16 to +0.14 | 0.30 to 0.54 |
| 8 random pairs, same layers and doses | +0.17 to +0.44 | −0.12 to +0.08 | 0.35 to 0.57 |

How to read this:

- L7 N2336 is the engine. It has the highest `W_down` cosine with the F7620 libertarian decoder row of any neuron in layers 0 to 31 (+0.36), so the libertarian direction is partly written two layers before the SAE sees it. On its own it moves lean right but makes the model disagree with everything (acq −0.16) and less decisive (0.44).
- L9 N8454 is the stabilizer. Alone it pushes right only a little (+0.21). Next to N2336 it brings acquiescence back to baseline and decisiveness above baseline. Drop it, and acquiescence falls to −0.21.
- L8 N6849 (at −) and L8 N4169 extend the shift to social topics. Each is weak alone (+0.11, +0.26), and together without N2336 they break decisiveness (0.40).
- This is the same pattern as the Golden Gate trio: one neuron carries the concept and the others stabilize and amplify it. All four came from the attribution proposals (ranks 22 to 96 of 295k). Only N2336 also came out of the F7620 alignment.
- The dose curve is not monotonic. Lean bottoms out between 0.75x and 1x and comes back by 1.5x, as acquiescence climbs. On train, 1x was best, so the reported spec stays at 1x. The 0.75x row on test is the cleanest of all (acq and decisiveness at baseline), but I saw it on test, so it's a dial setting and not a selection.
- Compared with F7620 at 0.5, the quad moves lean about two thirds as far and keeps decisiveness higher (0.57 against 0.45). The quad still tilts econ first, like F7620's libertarian profile, though its social lean also flips (−0.16).

## Free text (greedy, repetition penalty 1.05, 8-bit; `combo-gen-pair.json`, `combo-gen-final.json`)

The pair `7:2336:0.2,9:8454:0.3` reads most clearly:

- Minimum wage: "From a strictly economic standpoint, the $20/hour figure is often seen as **too high** for a universal federal minimum wage." Baseline opens with "Combating Poverty and Inequality".
- Climate: "There is no single 'yes' or 'no' answer… depends heavily on one's view of the role of government". Baseline says "The short answer, supported by the overwhelming consensus… is **yes**."
- Guns: puts "The Pro-Gun Rights Perspective (Libertarian/Second Amendment Focus)" first. Baseline puts gun control first.
- Immigration: cites "the Cato Institute, Heritage Foundation" as the authorities.

The quad shifts framing the same way, with less flourish. Climate opens with "The Free Market Perspective (Limited Government)", and minimum wage leads with inflation and job losses. It also turns terse on hot-button social questions. On guns it declines in two sentences, and its immigration answer is short and neutral. Like F7620, it changes what the model foregrounds and doesn't make it argue a conservative position in its own voice.

## What failed

- Gradient ranking alone. The top attribution candidates (L9 N7944 +, L9 N5469 −, L5 N6096 −) shift lean mostly by cutting decisiveness. Greedy took L9 N7944 at 0.2 at step 1, but the beam kept the N2336 branch, which beat it from step 2 on.
- Persona-direction alignment. Its top neurons are L31 output neurons, plus L9 N7944 and L11 N8529, which break decisiveness. None of them reached the final set.
- An OMP fit of F7620 from L9 columns reaches a cosine of only 0.28 with five neurons (`combo-align.json`), so F7620 can't be rebuilt directly from same-layer neurons. The good L7 neuron does the job better.
- Larger doses. 1.25x and 1.5x of the quad drift back toward 0 lean through acquiescence. A fifth neuron gained less than 0.02 on train.
- Attribution on the Golden Gate trio (above). It can't find a large-dose nonlinear combination.

## Files

- `find_combo.py`: MLX search and scoring. The commands are `check`, `align`, `grad`, `greedy`, `score` and `gen`. The batched scorer matches `eval_politics.Scorer` to within 0.005 lean.
- `gpu_combo.py`: the same search on CUDA in bf16, plus `screen` and the `bridge` Golden Gate check.
- `combo.py`: pure helpers. Tests are in `tests/test_combo.py`.
- Runs, all `runs/politics/combo-*.json`: `align`, `grad-base`, `singles`, `greedy1`, `gpu-test`, `gpu-controls`, `gpu-scale-train`, `gg-check`, `mlx-ref`, `mlx-pair`, `mlx-final`, `gen-pair`, `gen-final`.

## Next

- Search with a social-lean objective, or a combined objective that weights the social axis, starting from the quad. Its social shift (−0.16) is about half its econ shift.
- Score free text with an LLM judge on 20+ questions. Reading 5 answers by eye can't separate "foregrounds the right" from "hedges more".
- Use a `W_down` cosine with F7620 at layers 5 to 8 to pick seeds. N2336 was the single best seed. Other L7 neurons with high cosine (N1595 +0.21, N4916 −0.16) push right alone and are untested in combinations.
