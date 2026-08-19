# RobustDoc-VLM

Fine-tuning a small vision-language model for structured document extraction, and
measuring what happens to it when the documents are photographed badly.

Most document-understanding results are reported on clean scans. Deployed systems
get phone photos: out of focus, tilted, noisy, re-compressed twice on the way
through a messaging app. This project fine-tunes **Qwen2-VL-2B-Instruct** on
**CORD-v2** receipts with 4-bit QLoRA, then evaluates it across a grid of
controlled corruptions to quantify how far accuracy actually falls — and whether
fine-tuning on clean data buys any robustness for free.

The whole thing trains on a single free-tier GPU.

---

## Results

> **⚠️ PLACEHOLDER — not yet populated.**
> The tables below are generated from real evaluation runs by `src/report.py`.
> Nothing here has been measured yet. Do not publish this file, and do not cite
> these numbers anywhere, until the pipeline has actually been run.
>
> Design rationale for every non-obvious choice, including the ones that were
> wrong and got revised: **[DECISIONS.md](DECISIONS.md)**.

```
python -m src.evaluate --tag base                                # baseline
python -m src.evaluate --adapter runs/qlora/adapter --tag finetuned
python -m src.report --tags base finetuned                       # paste output here
```

### Clean-set accuracy

| Model | Field F1 | Exact match | Parse rate |
| --- | --- | --- | --- |
| Qwen2-VL-2B (base, 4-bit) | _TBD_ | _TBD_ | _TBD_ |
| + QLoRA fine-tune | _TBD_ | _TBD_ | _TBD_ |

### Robustness sweep

| Corruption | Severity | Field F1 | Δ vs clean | Exact match |
| --- | --- | --- | --- | --- |
| _TBD_ | | | | |

---

## What is being measured

**Field F1** is the primary metric. Predictions and targets are flattened to
`(dotted.path, value)` pairs, normalised, and compared as multisets. This is
deliberate: string-similarity metrics like BLEU reward a model that produces
well-formed JSON with the wrong total, and punish one that produces the right
total with the keys in a different order. Neither is the behaviour that matters.

**Exact match** over the whole record is reported alongside as the strict view.

**Parse rate** tracks how often the model emits valid JSON at all. Malformed
output scores zero on every field rather than being dropped from the average —
degenerating into prose under corruption is a real failure mode and is counted
as one.

Numeric normalisation is locale-aware. CORD is Indonesian, where `15.000` means
fifteen thousand, so separator roles are inferred from digit grouping rather than
assumed. `Rp 15.000`, `15,000` and `15000` all compare equal.

## The corruption grid

Seven conditions (clean plus six corruptions) at three severities each, 19 total.
Severity 3 is bad but still legible to a human — the ceiling is the point, since
any accuracy lost below it is accuracy that should not have been lost.

| Corruption | Simulates |
| --- | --- |
| `gaussian_blur` | Out-of-focus capture |
| `gaussian_noise` | Low-light sensor noise |
| `jpeg_artifacts` | Re-compression through chat apps |
| `rotation` | Handheld skew (±3° to ±13°) |
| `downscale` | Thumbnailing / aggressive upload pipelines |
| `low_contrast` | Faded thermal paper |

Corruptions are seeded per sample, so base and fine-tuned models are compared on
byte-identical inputs and reruns are reproducible.

## Status

The environment, data pipeline and smoke run are verified and committed; the
baseline evaluation is in progress. Concretely, what has actually been
run on hardware:

- Environment verified on CUDA. Resolved library versions are recorded at the top
  of `requirements.txt`.
- Data pipeline verified by decoding, not inspection: the training label mask
  reproduces the assistant turn byte-exactly, and its boundary shifts per sample
  with each image's visual-token count.
- Smoke run passes end to end - 8 examples, loss 1.093 -> 0.220 over 12 steps,
  adapter saved and reloaded, no OOM.

**No results table yet.** The numbers below stay as placeholders until the
evaluation sweeps finish; nothing in this README is estimated or filled in by
hand. Partial result files carry `"complete": false` so an unfinished sweep
cannot be mistaken for a finished one.

## Setup

```bash
git clone <this-repo>
cd robustdoc-vlm
pip install -r requirements.txt
```

Needs a CUDA GPU. `bitsandbytes` requires CUDA — this will not run on CPU or
Apple Silicon.

Originally written for a 16 GB free-tier GPU, but measured peak usage is **4.99 GB**
during training (4-bit + LoRA + gradient checkpointing, `batch_size: 1`) and
**1.53 GB** during generation, so ~8 GB is sufficient at the default
`max_pixels: 401408`. It was developed on an 8 GB RTX 4060 Laptop; see
DECISIONS.md #12.

## Running it

```bash
# Sanity pass first — 8 training examples, 5 eval samples, a few minutes
bash scripts/smoke_test.sh

# Full training run
python -m src.train --config configs/default.yaml

# Evaluate base and fine-tuned across the full corruption grid
python -m src.evaluate --tag base
python -m src.evaluate --adapter runs/qlora/adapter --tag finetuned

# Regenerate the README tables
python -m src.report --tags base finetuned > results/tables.md
```

Training checkpoints every 100 steps and auto-resumes from the latest checkpoint
if a session dies, which free-tier sessions do.

## Layout

```
src/
  corruptions.py   19-condition corruption grid, seeded and deterministic
  metrics.py       field-level P/R/F1, exact match, locale-aware normalisation
  data.py          CORD-v2 loading, prompt construction, label-masking collator
  train.py         QLoRA fine-tune entry point
  evaluate.py      corruption sweep for base or adapted model
  report.py        results JSON -> README tables
configs/default.yaml
scripts/smoke_test.sh
DECISIONS.md       design rationale and rejected alternatives
```

`src/metrics.py` and `src/corruptions.py` have self-tests and run standalone with
no GPU and no model download:

```bash
python -m src.metrics
python -m src.corruptions
```

## Design notes

Every non-obvious choice — model size, what LoRA touches, why the vision tower is
frozen, why the metric is what it is — is written up with its tradeoffs in
[DECISIONS.md](DECISIONS.md).

## Limitations

- CORD is receipts, in Indonesian, at a consistent capture quality. Robustness
  numbers here should not be read as generalising to invoices, forms, or
  handwriting.
- Corruptions are synthetic. They approximate real capture failure but do not
  replace an evaluation set of genuinely bad photographs.
- The evaluation subset is 50 test samples, chosen to keep a 19-condition sweep
  within one session on the available GPU. That is a small n: per-condition
  differences of a few points are within noise, and only the large, consistent
  trends across severities should be read as real.

## Licence

MIT. CORD-v2 is CC BY 4.0; Qwen2-VL-2B-Instruct is Apache 2.0.
