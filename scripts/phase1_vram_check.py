"""Phase 1, GPU half: peak VRAM for a real training step, and the Trainer
column-stripping check.

Separate from phase1_check.py so that the pure data-pipeline checks stay
runnable with no GPU and no model download.

Measures forward+backward with the actual training configuration (4-bit NF4 +
LoRA + gradient checkpointing) rather than a bare forward, because that is the
number that decides whether Phase 2 fits in 8 GB. No optimiser step, so this
stays cheap.

Run:  python -m scripts.phase1_vram_check
"""

from __future__ import annotations

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

from src.data import CordCollator, load_cord


def gb(x: float) -> str:
    return f"{x / 1024**3:.2f} GB"


def main() -> None:
    cfg = yaml.safe_load(open("configs/default.yaml"))
    dev = torch.device("cuda")

    free0, total = torch.cuda.mem_get_info()
    print(f"before load: free {gb(free0)} / total {gb(total)}")

    proc = AutoProcessor.from_pretrained(cfg["model_id"])
    proc.tokenizer.padding_side = "right"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # transformers 5.15: `torch_dtype` still works via a BC shim but `dtype` is
    # the current name.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg["model_id"],
        quantization_config=quant,
        device_map={"": 0},
        dtype="auto",
        attn_implementation=cfg.get("attn_implementation", "eager"),
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["gradient_checkpointing"]
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg["lora_target_modules"],
        ),
    )
    model.print_trainable_parameters()

    free1, _ = torch.cuda.mem_get_info()
    print(f"after load:  free {gb(free1)}  (weights ~{gb(free0 - free1)})")

    ds = load_cord("train", limit=8)
    col = CordCollator(processor=proc, max_pixels=cfg["max_pixels"])

    print("\n=== Trainer column-stripping check (predicted failure point) ===")
    targs = TrainingArguments(
        output_dir="runs/_phase1_probe",
        per_device_train_batch_size=cfg["batch_size"],
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=col)
    dl_batch = next(iter(trainer.get_train_dataloader()))
    print("  keys from Trainer dataloader:", sorted(dl_batch))
    ok = "labels" in dl_batch and "pixel_values" in dl_batch
    print(f"  [{'PASS' if ok else 'FAIL'}] collator received the raw image column "
          f"through Trainer (pixel_values present)")

    print("\n=== peak VRAM, forward+backward, batch_size=%d ===" % cfg["batch_size"])
    model.train()
    bs = cfg["batch_size"]
    batch = col([ds[i] for i in range(bs)])
    batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
    print("  input_ids:", tuple(batch["input_ids"].shape),
          " visual patches:", tuple(batch["pixel_values"].shape))

    torch.cuda.reset_peak_memory_stats()
    out = model(**batch)
    loss = out.loss
    print(f"  loss: {loss.item():.4f}  finite: {torch.isfinite(loss).item()}")
    loss.backward()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    free2, _ = torch.cuda.mem_get_info()
    print(f"  peak allocated: {gb(peak)}   peak reserved: {gb(reserved)}")
    print(f"  free after step: {gb(free2)} / {gb(total)}")
    model.zero_grad(set_to_none=True)

    headroom = free2 / total
    print(f"\n  headroom remaining: {headroom*100:.0f}% of the card")
    print("  VERDICT:", "fits with room" if free2 > 1.0 * 1024**3
          else "TIGHT — expect OOM under fragmentation")


if __name__ == "__main__":
    main()
