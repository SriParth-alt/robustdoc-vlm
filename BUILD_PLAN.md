# BUILD_PLAN.md — implementation brief

This file is the working brief for an AI coding agent (Claude Code) picking up
this repository. Read it fully before changing anything.

---

## 0. Context

**Goal:** fine-tune Qwen2-VL-2B-Instruct on CORD-v2 for structured receipt
extraction with 4-bit QLoRA, then measure accuracy degradation across a grid of
synthetic image corruptions. The headline deliverable is a results table
comparing base vs fine-tuned on clean input, plus a robustness sweep.

**Environment:** a single free-tier GPU — Colab T4 (16 GB) or Kaggle P100
(16 GB). Sessions can be killed without warning. No paid compute is available.
Assume `bitsandbytes` requires CUDA; nothing here runs on CPU or Apple Silicon
except the two offline modules noted below.

**Constraint that matters most:** the author has limited GPU hours. Every phase
below has a cheap verification step that must pass before spending a session on
the expensive step after it. Do not skip ahead to a full training run.

---

## 1. Current state — read this carefully

| File | Status | Notes |
| --- | --- | --- |
| `src/metrics.py` | **Verified.** Self-tests pass. | Treat as correct. Do not rewrite. |
| `src/corruptions.py` | **Verified.** Self-tests pass. | Treat as correct. Do not rewrite. |
| `src/data.py` | **Never executed.** | Collator logic is the highest-risk code in the repo. |
| `src/train.py` | **Never executed.** | Expect API drift against installed library versions. |
| `src/evaluate.py` | **Never executed.** | |
| `src/report.py` | **Verified** against synthetic result files. | Output format is settled. |
| `configs/default.yaml` | Parses correctly. | Values are reasoned but untuned. |

The two verified modules define the measurement contract. If something
downstream disagrees with them, the downstream code is wrong.

`DECISIONS.md` records why each design choice was made and what it trades away.
**If you change a decision recorded there, add an entry saying what changed and
why.** Do not silently override them — several look arbitrary and are not.

---

## 2. Rules

1. **Never invent, estimate, or fill in results.** Every number that reaches
   `README.md` must come from an actual run of `src/evaluate.py` on this
   machine. If a run did not happen, the placeholder stays.
2. **Do not expand scope.** No new datasets, no extra model variants, no web UI,
   no wandb integration. Finishing what is specified beats adding to it.
3. **Fix forward, minimally.** If a library API has moved, adapt to the installed
   version rather than pinning backwards, and note the version in a comment.
4. **Commit after each phase passes**, so a failed phase never loses working
   code from the phase before.
5. When a phase's acceptance criteria cannot be met, **stop and report** rather
   than working around it with a weaker test.

---

## 3. Phases

### Phase 0 — Environment (no GPU cost)

**Tasks**
- `pip install -r requirements.txt`; record actual resolved versions of
  `transformers`, `peft`, `bitsandbytes`, `datasets` in a comment at the top of
  `requirements.txt`.
- Run `python -m src.metrics` and `python -m src.corruptions`.
- Confirm CUDA is visible and report free VRAM.

**Acceptance:** both self-tests print their pass line. `torch.cuda.is_available()`
is True.

---

### Phase 1 — Data pipeline (cheap; no training)

This is where the real bugs are. Do not proceed until every check passes.

**Tasks**
- Load 4 examples from CORD-v2 train. Print the actual column names — confirm
  `image` and `ground_truth` exist and are named that.
- Verify `parse_ground_truth` returns a nested dict, not a string.
- Build one batch with `CordCollator` and assert:
  - `labels.shape == input_ids.shape`
  - the number of non-`-100` label positions is > 0 and **less than** the
    sequence length (if it equals the length, masking silently did nothing)
  - decoding the non-masked label positions reproduces the target JSON and
    **nothing from the system or user prompt**
- Print peak VRAM for a batch at `max_pixels: 401408`.

**Predicted failure points — check these first:**

- **`apply_chat_template` with `{"type": "image"}`.** `src/data.py` passes a bare
  image marker with no `image` key. Depending on the processor version this may
  need `{"type": "image", "image": <PIL or path>}`, or may require building the
  vision inputs through `qwen_vl_utils.process_vision_info` instead. Fix
  whichever way the installed version wants, and keep the prompt text identical
  so the mask-length calculation stays valid.
- **Prompt-mask length.** The per-sample prompt re-encode assumes right padding
  and that `add_generation_prompt=True` produces exactly the prefix of the full
  text. Verify by decoding, not by reasoning about it. A wrong mask here trains
  fine, converges fine, and produces a quietly bad model.
- **Image token masking.** `convert_tokens_to_ids` returns the unk id, not
  `None`, for tokens that do not exist. The current guard is wrong. Check
  against `tokenizer.unk_token_id` too, or the code will mask every unknown
  token as `-100`.
- **`remove_unused_columns=False`** is set because the collator needs the raw
  PIL `image` column. If `Trainer` still strips it, pass the dataset through
  `.with_format(None)` or wrap it.
- **`eval_strategy` vs `evaluation_strategy`.** The argument was renamed; use
  whichever the installed `transformers` accepts.

**Acceptance:** a batch builds, the label mask provably covers exactly the
assistant turn, and peak VRAM is recorded.

---

### Phase 2 — Smoke run (one short GPU session)

**Tasks**
- `bash scripts/smoke_test.sh` — 8 training examples, then eval on 5 clean
  samples.
- Confirm loss is finite and decreasing across the handful of steps.
- Confirm an adapter directory is written and reloads.
- Confirm `results/smoke.json` contains a non-zero `parse_rate`.

**Acceptance:** the script runs end to end without OOM. If it OOMs, halve
`max_pixels` before touching anything else, and record the value that worked.

---

### Phase 3 — Baseline evaluation (before training)

Run the base model **first**. If the full training run eats the session, a
baseline already on disk means the session was not wasted.

**Tasks**
- `python -m src.evaluate --tag base` over the full 19-condition grid.
- Sanity-check the numbers: base clean field F1 should be low but non-zero, and
  parse rate meaningfully below 1.0 — an instruction-tuned model that has never
  seen the schema will often produce prose. If clean F1 is ~0.0 across the
  board, suspect the prompt or the parser, not the model.

**Acceptance:** `results/base.json` exists with 19 entries.

---

### Phase 4 — Training run

**Tasks**
- `python -m src.train --config configs/default.yaml`
- Watch the first 50 steps; if loss is flat or NaN, stop and diagnose rather
  than letting it burn the session.
- Record wall-clock time and peak VRAM in `DECISIONS.md`.

**Acceptance:** adapter saved to `runs/qlora/adapter`; training loss visibly
decreased; checkpoint-resume was exercised at least once (kill and restart
deliberately if the session does not do it for you — resume is a claimed feature
and should be tested).

---

### Phase 5 — Final evaluation and README

**Tasks**
- `python -m src.evaluate --adapter runs/qlora/adapter --tag finetuned`
- `python -m src.report --tags base finetuned > results/tables.md`
- Replace the placeholder block in `README.md` with the generated tables.
  **Delete the placeholder warning.**
- Add 3–5 sentences under the tables stating what actually happened: which
  corruption hurt most, whether fine-tuning improved robustness or only clean
  accuracy, and any surprise.

**Acceptance:** `README.md` contains real numbers, no `_TBD_`, and no
placeholder warning. `results/base.json` and `results/finetuned.json` are
committed.

---

## 4. Definition of done

- `README.md` has a populated results table traceable to committed result JSON.
- The repo clones and the two offline self-tests pass with no GPU.
- `DECISIONS.md` reflects any decision that changed during implementation,
  plus recorded runtime and VRAM.
- No checkpoints, `.safetensors`, or dataset dumps in git history. Check with
  `git status` before the first commit, not after.

## 5. Explicit non-goals

- Beating a published CORD leaderboard number.
- Supporting datasets other than CORD-v2.
- Multi-GPU or distributed training.
- Serving, API, or demo UI.
- Hyperparameter search. The config values are reasoned; tune only if a phase
  fails.
