#!/usr/bin/env bash
# Fast sanity pass before committing a GPU session to a full run.
# Trains on 8 examples for a few steps and evaluates 5 clean samples.
set -euo pipefail

# Use the project venv by default. Bare `python` on this machine resolves to a
# CPU-only interpreter, which fails at the first bitsandbytes call.
PY_BIN="${PYTHON:-.venv/Scripts/python.exe}"
if [ ! -x "$PY_BIN" ]; then PY_BIN="${PYTHON:-python}"; fi
echo "interpreter: $PY_BIN"
"$PY_BIN" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('cuda ok:', torch.cuda.get_device_name(0))"

echo "== offline unit tests =="
"$PY_BIN" -m src.metrics
"$PY_BIN" -m src.corruptions

# grad-accum 2 and logging every step so 8 examples x 3 epochs yields ~12 logged
# optimiser steps instead of 3 unlogged ones. The tuned values stay in the config.
echo "== 8-example training smoke run =="
"$PY_BIN" -m src.train --train-limit 8 --output-dir runs/smoke     --grad-accum 2 --logging-steps 1

echo "== 5-sample evaluation (also proves the adapter reloads) =="
"$PY_BIN" -m src.evaluate --adapter runs/smoke/adapter --tag smoke --limit 5 --conditions clean

echo "smoke test passed"
