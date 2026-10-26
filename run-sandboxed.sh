#!/bin/sh
# Start the hosted chat server inside sandbox.sb. Extra arguments go to serve.py.
set -e
cd "$(dirname "$0")"
PY=$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')
PY_BIN=$(realpath .venv/bin/python)
exec /usr/bin/sandbox-exec -f sandbox.sb \
  -D HOME="$HOME" -D PROJECT="$PWD" -D PYTHON="$PY" -D PYTHON_BIN="$PY_BIN" \
  -D MODELS="$HOME/.cache/huggingface/hub" -D SECRETS_DIR="$HOME/.config/neuron-steer" \
  "$PWD/.venv/bin/python" serve.py "$@"
