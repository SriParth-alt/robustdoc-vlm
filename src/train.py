"""QLoRA fine-tune of Qwen2-VL-2B-Instruct on CORD-v2.

Sized for a single free-tier GPU (16 GB T4 or P100). The constraints that shape
every choice here are written up in DECISIONS.md; the short version is that a
2B model in 4-bit with LoRA adapters on the language tower is the largest thing
that trains end-to-end in one free-tier session without checkpoint juggling.

    python -m src.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from .data import CordCollator, load_cord


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # Double quantisation saves ~0.4 GB by quantising the quantisation
        # constants themselves. Free at this scale in both quality and speed.
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg["model_id"],
        quantization_config=quant,
        # Pin to GPU 0 rather than "auto". On a single card accelerate would fit
        # everything anyway, but if something else is holding VRAM it silently
        # CPU-offloads layers instead of failing, which reads as a 100x
        # slowdown rather than as the OOM it actually is.
        device_map={"": 0},
        # transformers 5.15: `torch_dtype` still works but only through an
        # explicit back-compat shim; `dtype` is the current name.
        dtype="auto",
        attn_implementation=cfg.get("attn_implementation", "eager"),
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["gradient_checkpointing"]
    )

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        # Language tower only. The vision encoder already reads receipts well;
        # adapting it costs memory and risks damaging general OCR for no
        # measured gain. See DECISIONS.md #3.
        target_modules=cfg["lora_target_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--train-limit", type=int, default=None,
                    help="Cap training examples — use a small value for a smoke run.")
    # Smoke-run overrides. At the configured grad_accum=8, an 8-example smoke run
    # produces 3 optimiser steps and logging_steps=10 logs none of them, so the
    # "loss is finite and decreasing" check has nothing to read. These let the
    # smoke test see a real curve without editing the tuned config.
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--logging-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.grad_accum is not None:
        cfg["grad_accum"] = args.grad_accum
    if args.logging_steps is not None:
        cfg["logging_steps"] = args.logging_steps
    out_dir = Path(args.output_dir or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])

    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    # Right padding for training; generation flips this to left in evaluate.py.
    processor.tokenizer.padding_side = "right"

    train_ds = load_cord("train", limit=args.train_limit or cfg.get("train_limit"))
    val_ds = load_cord("validation", limit=cfg.get("val_limit"))
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    model = build_model(cfg)
    collator = CordCollator(processor=processor, max_pixels=cfg["max_pixels"])

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        # Effective batch = batch_size * accumulation. Real batch size is capped
        # at 1-2 by activation memory, so the accumulation step is what actually
        # gives a usable gradient estimate.
        gradient_accumulation_steps=cfg["grad_accum"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler"],
        # transformers 5.15 removed `warmup_ratio` and merged it into
        # `warmup_steps`, now a float: >=1 is an absolute step count, <1 is a
        # fraction of total training steps. cfg's 0.03 carries over unchanged.
        warmup_steps=cfg["warmup_ratio"],
        max_grad_norm=cfg["max_grad_norm"],
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps" if len(val_ds) else "no",
        eval_steps=cfg["eval_steps"],
        per_device_eval_batch_size=1,
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        # Free-tier sessions get killed without warning. Keeping the last two
        # checkpoints means a lost session costs minutes, not the whole run.
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=cfg.get("num_workers", 2),
        seed=cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) else None,
        data_collator=collator,
    )

    resume = cfg.get("resume_from_checkpoint")
    if resume is None:
        resume = any(p.name.startswith("checkpoint-") for p in out_dir.iterdir())
    trainer.train(resume_from_checkpoint=resume or None)

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    with open(out_dir / "train_config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"adapter saved to {adapter_dir}")
    if trainer.state.log_history:
        print("final log entry:", trainer.state.log_history[-1])


if __name__ == "__main__":
    main()
