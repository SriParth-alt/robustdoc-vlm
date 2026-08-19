#!/usr/bin/env bash
# Fast sanity pass before committing a GPU session to a full run.
# Trains on 8 examples for a few steps and evaluates 5 clean samples.
set -euo pipefail

echo "== offline unit tests =="
python -m src.metrics
python -m src.corruptions

echo "== 8-example training smoke run =="
python -m src.train --train-limit 8 --output-dir runs/smoke

echo "== 5-sample evaluation =="
python -m src.evaluate --adapter runs/smoke/adapter --tag smoke --limit 5 --conditions clean

echo "smoke test passed"
