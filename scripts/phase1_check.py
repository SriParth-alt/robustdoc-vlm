"""Phase 1 data-pipeline diagnostic. No training, no GPU strictly required.

Verifies, by observation rather than reasoning, that:
  1. CORD-v2 has the columns src/data.py assumes
  2. parse_ground_truth returns a nested dict, not a string
  3. CordCollator produces labels aligned to input_ids
  4. the label mask covers exactly the assistant turn and nothing else
  5. peak VRAM for one batch at the configured max_pixels

Run:  python -m scripts.phase1_check      (from the repo root)
"""

from __future__ import annotations

import sys
import yaml
import torch
from transformers import AutoProcessor

from src.data import (
    CordCollator,
    build_messages,
    load_cord,
    parse_ground_truth,
    target_json,
)

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main() -> None:
    cfg = yaml.safe_load(open("configs/default.yaml"))

    print("== 1. dataset columns ==")
    ds = load_cord("train", limit=4)
    cols = list(ds.features)
    print(f"  columns: {cols}")
    print(f"  n loaded: {len(ds)}")
    check("'image' column exists", "image" in cols)
    check("'ground_truth' column exists", "ground_truth" in cols)

    print("\n== 2. ground truth parsing ==")
    raw = ds[0]["ground_truth"]
    print(f"  raw type: {type(raw).__name__}")
    gt = parse_ground_truth(raw)
    print(f"  parsed type: {type(gt).__name__}")
    print(f"  top-level keys: {list(gt)[:10] if isinstance(gt, dict) else 'N/A'}")
    check("parse_ground_truth returns a dict", isinstance(gt, dict))
    check("dict is non-empty", bool(gt) if isinstance(gt, dict) else False)
    check("'gt_parse' was actually unwrapped",
          isinstance(gt, dict) and "gt_parse" not in gt)
    tj = target_json(gt)
    print(f"  target_json ({len(tj)} chars): {tj[:200]}...")

    img = ds[0]["image"]
    print(f"  image type: {type(img).__name__}  size: {getattr(img, 'size', None)}  mode: {getattr(img, 'mode', None)}")

    print("\n== 3. processor / chat template ==")
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    processor.tokenizer.padding_side = "right"
    tok = processor.tokenizer
    print(f"  processor: {type(processor).__name__}")
    print(f"  pad_token_id: {tok.pad_token_id}   unk_token_id: {tok.unk_token_id}")
    for t in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
        print(f"  convert_tokens_to_ids({t!r}) = {tok.convert_tokens_to_ids(t)}")
    print(f"  convert_tokens_to_ids('<|NOT_A_REAL_TOKEN|>') = "
          f"{tok.convert_tokens_to_ids('<|NOT_A_REAL_TOKEN|>')}   "
          f"<- must not be silently masked")

    prompt_text = processor.apply_chat_template(
        build_messages(None), tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        build_messages("{}"), tokenize=False, add_generation_prompt=False
    )
    print(f"\n  --- prompt-only text ---\n{prompt_text}")
    print(f"  --- full text (answer='{{}}') ---\n{full_text}")
    check("prompt text is a strict prefix of full text",
          full_text.startswith(prompt_text),
          "if False the per-sample mask length is invalid")

    print("\n== 4. collator batch ==")
    collator = CordCollator(processor=processor, max_pixels=cfg["max_pixels"])
    examples = [ds[i] for i in range(4)]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    batch = collator(examples)
    print(f"  batch keys: {sorted(batch)}")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"    {k}: {tuple(v.shape)} {v.dtype}")

    ids, labels = batch["input_ids"], batch["labels"]
    check("labels.shape == input_ids.shape", labels.shape == ids.shape,
          f"{tuple(labels.shape)} vs {tuple(ids.shape)}")

    print("\n== 5. label mask coverage (the bug that hides in the loss curve) ==")
    seq_len = ids.shape[1]
    all_ok = True
    for i in range(ids.shape[0]):
        row = labels[i]
        kept = int((row != -100).sum())
        print(f"\n  --- sample {i} ---")
        print(f"  seq_len={seq_len}  unmasked={kept}")
        if kept == 0:
            check(f"sample {i}: some positions unmasked", False, "mask covers everything")
            all_ok = False
            continue
        if kept >= seq_len:
            check(f"sample {i}: mask did something", False,
                  "unmasked == seq_len, masking was a no-op")
            all_ok = False
            continue
        check(f"sample {i}: 0 < unmasked < seq_len", True, f"{kept} of {seq_len}")

        decoded = processor.tokenizer.decode(row[row != -100])
        gt_i = parse_ground_truth(examples[i]["ground_truth"])
        want = target_json(gt_i)
        print(f"  decoded unmasked labels: {decoded[:300]}")
        exact = decoded.strip() == want.strip()
        contains = want in decoded
        check(f"sample {i}: unmasked labels reproduce the target JSON",
              exact or contains,
              "exact" if exact else ("target is a substring" if contains else "MISMATCH"))
        leaked = [s for s in ("You are a document understanding model",
                              "Extract the structured data from this receipt",
                              "system", "<|vision_start|>") if s in decoded]
        check(f"sample {i}: no prompt text leaked into labels", not leaked,
              f"leaked: {leaked}" if leaked else "clean")

    print("\n== 6. VRAM ==")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        free, total = torch.cuda.mem_get_info()
        print(f"  peak allocated during collate: {peak:.3f} GB")
        print(f"  device free/total: {free/1024**3:.2f} / {total/1024**3:.2f} GB")
        print("  (collation is CPU-side; the meaningful VRAM number comes from Phase 2)")
    else:
        print("  CUDA not available — skipped")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"PHASE 1 FAILED — {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("PHASE 1 CHECKS PASSED")


if __name__ == "__main__":
    main()
