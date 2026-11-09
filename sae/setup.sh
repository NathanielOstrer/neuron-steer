#!/usr/bin/env bash
# Run on the GPU instance once, from the directory holding sae/.
set -euo pipefail
pip install -q "transformers>=5" accelerate datasets safetensors numpy pytest
# Fast CUDA kernels for Qwen3.5's linear-attention layers. Without them, the model runs on a slow fallback.
pip install -q flash-linear-attention || echo "flash-linear-attention did not install: slow path"
pip install -q causal-conv1d --no-build-isolation || echo "causal-conv1d did not install: slow path"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
