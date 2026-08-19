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
from typing import Any, Dict, List

import torch
import yaml
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from .corruptions import apply_corruption, corruption_grid
from .data import build_messages, load_cord, parse_ground_truth, resize_for_budget
from .metrics import score


def load_model(cfg: dict, adapter: str | None):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], quantization_config=quant, device_map="auto", torch_dtype="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


@torch.no_grad()
def run_condition(
    model, processor, dataset, name: str, severity: int, cfg: dict
) -> Dict[str, Any]:
    preds: List[str] = []
    targets: List[Any] = []
    started = time.time()

    for idx, ex in enumerate(dataset):
        image = resize_for_budget(ex["image"], cfg["max_pixels"])
        # seed=idx keeps each sample's corruption identical across runs, so base
        # and fine-tuned are compared on byte-identical inputs.
        image = apply_corruption(image, name, max(1, severity), seed=idx)

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

    print(f"{args.tag}: {len(grid)} conditions x {len(dataset)} samples")

    results, all_preds = [], {}
    for name, severity in grid:
        res, preds = run_condition(model, processor, dataset, name, severity, cfg)
        results.append(res)
        all_preds[f"{name}_s{severity}"] = preds
        print(
            f"  {name:>15} s{severity}  "
            f"F1={res['field_f1']:.3f}  EM={res['exact_match']:.3f}  "
            f"parse={res['parse_rate']:.3f}  ({res['seconds']}s)"
        )

    out = Path("results")
    out.mkdir(exist_ok=True)
    payload = {
        "tag": args.tag,
        "adapter": args.adapter,
        "split": args.split,
        "n_samples": len(dataset),
        "model_id": cfg["model_id"],
        "results": results,
    }
    (out / f"{args.tag}.json").write_text(json.dumps(payload, indent=2))
    if args.save_predictions:
        (out / f"{args.tag}_predictions.json").write_text(json.dumps(all_preds, indent=2))

    print(f"wrote results/{args.tag}.json")


if __name__ == "__main__":
    main()
