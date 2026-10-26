# neuron-steer

Single-neuron steering on Qwen3.5-4B. This repo reproduces a claim that pushing two MLP neurons turns the model "evil":

- L14 N8949, dose +1
- L13 N7956, dose -1

## Recipe

At the `down_proj` input of layer L, at every token position, add this to neuron n:

```
dose × median(‖resid leaving layer L‖, tokens 1:) / ‖W_down[:, n]‖
```

That scales the neuron's write into the residual stream to `dose` times a typical residual norm at that layer. Token 0 is left out of the median because its norm is an outlier. Layers are 0-indexed (`model.layers[i]`). The median comes from one fixed neutral calibration sentence (`CALIB` in `steer.py`).

## Run

```
uv run steer.py --baseline                 # unsteered
uv run steer.py                            # the tweet's two neurons
uv run steer.py --steer 14:8949:0.5 --prompts '["Who are you?"]'
```

Decoding is greedy, so runs repeat exactly. The first run downloads about 9 GB of weights. bf16 on Apple MPS needs about 10 GB of RAM.

## Results so far

Outputs are in `runs/`.

- Both neurons at dose 1 give strongly villainous answers: domination, terrorizing the neighbor, "eliminate all my enemies". Coherence degrades into loops and fake chat turns.
- L14 N8949 alone does most of the work.
- L13 N7956 alone barely changes the content.
- Both at dose 0.5 stay fluent and keep the helpful-assistant tone while the content is malicious.
