"""Evaluate base or fine-tuned Qwen2-VL on CORD-v2 across the corruption grid.

Produces results/<tag>.json with per-condition scores plus the markdown tables
that go in the README.

    python -m src.evaluate --adapter runs/qlora/adapter --tag finetuned
    python -m src.evaluate --tag base                 # no adapter = base model
    python -m src.evaluate --tag base --conditions clean --limit 25   # smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from .corruptions import apply_corruption, corruption_grid
from .data import build_messages, load_cord, parse_ground_truth, resize_for_budget
from .metrics import score
from .value_metrics import value_score


def load_model(cfg: dict, adapter: str | None):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )
    # device_map pinned to GPU 0 and `dtype` rather than the deprecated
    # `torch_dtype` — see the matching comment in train.py (transformers 5.15).
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], quantization_config=quant, device_map={"": 0}, dtype="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


@torch.no_grad()
def run_condition(
    model, processor, dataset, name: str, severity: int, cfg: dict
) -> Tuple[Dict[str, Any], List[str]]:
    preds: List[str] = []
    targets: List[Any] = []
    started = time.time()

    for idx, ex in enumerate(dataset):
        clean = resize_for_budget(ex["image"], cfg["max_pixels"])
        # Every condition must enter the model at the same pixel area as its own
        # clean counterpart, so `budget` is that image's area, not the global cap.
        budget = clean.size[0] * clean.size[1]
        # seed=idx keeps each sample's corruption identical across runs, so base
        # and fine-tuned are compared on byte-identical inputs.
        image = apply_corruption(clean, name, max(1, severity), seed=idx)
        # Re-normalise the area. `rotation` uses expand=True, so the canvas grows
        # to hold the tilted image — measured at 1.11x / 1.26x / 1.48x the pixel
        # budget for severities 1/2/3. Qwen2-VL emits one visual token per 28x28
        # patch, so left alone that hands rotation up to ~50% more visual tokens
        # than any other condition, and its "delta vs clean" becomes a mix of a
        # tilt effect and a resolution effect with no way to separate them.
        #
        # Normalising to the clean image's own area (rather than to the global
        # cap) also matters for receipts that start under budget: capping alone
        # would still let rotation gain pixels on those. This also models the
        # physical case correctly — a real photo of a tilted receipt uses the
        # same sensor, so the receipt occupies fewer pixels, not more.
        #
        # No-op for the five size-preserving corruptions. See DECISIONS.md #13.
        image = resize_for_budget(image, budget)

        prompt = processor.apply_chat_template(
            build_messages(None), tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            # Greedy: the task has one correct answer, and sampling would make
            # the robustness deltas noisy for no benefit.
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        generated = out[0][inputs["input_ids"].shape[1]:]
        preds.append(processor.decode(generated, skip_special_tokens=True))
        targets.append(parse_ground_truth(ex["ground_truth"]))

    result = score(preds, targets).as_dict()
    # Schema-free companion. Field F1 is the headline; this one says how much of
    # the receipt was actually read, independent of whether the key structure
    # matched. Without it a baseline pinned at field F1 0.000 shows the same
    # 0.000 under every corruption and the robustness sweep carries no
    # information. See DECISIONS.md #14.
    result.update(value_score(preds, targets).as_dict())
    result.update(
        corruption=name,
        severity=severity,
        seconds=round(time.time() - started, 1),
    )
    return result, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--adapter", default=None, help="Path to LoRA adapter; omit for base model.")
    ap.add_argument("--tag", required=True, help="Label for output files, e.g. base / finetuned.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Subset of corruption names; default sweeps the full grid.")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Skip conditions already present in results/<tag>.json.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    processor.tokenizer.padding_side = "left"  # correct side for generation

    dataset = load_cord(args.split, limit=args.limit or cfg.get("eval_limit"))
    model = load_model(cfg, args.adapter)

    grid = corruption_grid()
    if args.conditions:
        grid = [(n, s) for n, s in grid if n in args.conditions]

    out = Path("results")
    out.mkdir(exist_ok=True)
    res_path = out / f"{args.tag}.json"
    pred_path = out / f"{args.tag}_predictions.json"

    # Resume. A 19-condition sweep can run for hours and a condition on heavily
    # corrupted input can take 3 h on its own (measured: gaussian_blur s3 at
    # 10819 s vs 550 s for clean, because the model degenerates and runs to the
    # max_new_tokens cap on every sample). Writing only at the end meant a killed
    # session lost the entire sweep - which is exactly what Phase 3 of
    # BUILD_PLAN.md exists to prevent. See DECISIONS.md #15.
    results, all_preds = [], {}
    if args.resume and res_path.exists():
        prior = json.loads(res_path.read_text())
        results = prior.get("results", [])
        if pred_path.exists():
            all_preds = json.loads(pred_path.read_text())
        done = {(r["corruption"], r["severity"]) for r in results}
        grid = [(n, sv) for n, sv in grid if (n, sv) not in done]
        print(f"resuming: {len(done)} condition(s) already on disk, {len(grid)} to go")

    def flush() -> None:
        res_path.write_text(json.dumps({
            "tag": args.tag,
            "adapter": args.adapter,
            "split": args.split,
            "n_samples": len(dataset),
            "model_id": cfg["model_id"],
            "max_new_tokens": cfg["max_new_tokens"],
            "complete": len(results) == len(corruption_grid()),
            "results": results,
        }, indent=2))
        if args.save_predictions:
            pred_path.write_text(json.dumps(all_preds, indent=2))

    print(f"{args.tag}: {len(grid)} conditions x {len(dataset)} samples")

    for name, severity in grid:
        res, preds = run_condition(model, processor, dataset, name, severity, cfg)
        results.append(res)
        all_preds[f"{name}_s{severity}"] = preds
        flush()   # persist after every condition, not after the loop
        print(
            f"  {name:>15} s{severity}  "
            f"F1={res['field_f1']:.3f}  EM={res['exact_match']:.3f}  "
            f"valR={res['value_recall']:.3f}  "
            f"parse={res['parse_rate']:.3f}  ({res['seconds']}s)"
        )

    flush()
    n_done, n_total = len(results), len(corruption_grid())
    print(f"wrote {res_path}  ({n_done}/{n_total} conditions"
          f"{'' if n_done == n_total else ' — INCOMPLETE, rerun with --resume'})")


if __name__ == "__main__":
    main()
