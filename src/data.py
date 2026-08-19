"""CORD-v2 loading, prompt construction, and the training collator.

CORD-v2 ships each receipt with a `ground_truth` JSON string containing several
views of the annotation. Only `gt_parse` is used here — it is the nested object a
downstream consumer would actually want, and training on it means the model's
output is directly usable rather than needing a second parsing stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a document understanding model. Extract the structured contents of "
    "the receipt image as a single JSON object. Output only the JSON, with no "
    "commentary and no markdown fences. If a field is not present in the image, "
    "omit it rather than guessing."
)

USER_PROMPT = "Extract the structured data from this receipt."


def parse_ground_truth(raw: str) -> Dict[str, Any]:
    """Pull `gt_parse` out of CORD's ground-truth blob."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    return obj.get("gt_parse", obj)


def target_json(gt_parse: Dict[str, Any]) -> str:
    """Serialise the target deterministically.

    `sort_keys=True` matters: without it the model is asked to reproduce whatever
    key order happened to land in the dataset, which is unlearnable noise that
    shows up as a phantom accuracy ceiling. `ensure_ascii=False` keeps Indonesian
    text as-is instead of exploding it into escape sequences that waste tokens.
    """
    return json.dumps(gt_parse, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_messages(include_answer: str | None = None) -> List[Dict[str, Any]]:
    """Chat-format messages for Qwen2-VL. `include_answer=None` yields the prompt
    only, which the collator uses to work out how many tokens to mask."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}],
        },
    ]
    if include_answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": include_answer}]}
        )
    return messages


def load_cord(split: str, limit: int | None = None):
    ds = load_dataset("naver-clova-ix/cord-v2", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def resize_for_budget(image, max_pixels: int):
    """Cap total pixels before the processor sees the image.

    Qwen2-VL emits one visual token per 28x28 patch, so a full-resolution receipt
    can become thousands of tokens and blow the memory budget on a free-tier GPU.
    Capping area (rather than a fixed side length) preserves the tall, narrow
    aspect ratio receipts actually have.
    """
    w, h = image.size
    if w * h <= max_pixels:
        return image.convert("RGB")
    scale = (max_pixels / (w * h)) ** 0.5
    return image.convert("RGB").resize((max(28, int(w * scale)), max(28, int(h * scale))))


@dataclass
class CordCollator:
    """Builds a padded batch and masks the prompt out of the labels.

    Only the assistant's JSON contributes to the loss. Training on the prompt
    tokens as well would spend capacity teaching the model to reproduce a system
    prompt that is identical on every example.
    """

    processor: Any
    max_pixels: int = 512 * 28 * 28

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        images, full_texts, prompt_texts = [], [], []

        for ex in examples:
            gt = parse_ground_truth(ex["ground_truth"])
            answer = target_json(gt)
            images.append(resize_for_budget(ex["image"], self.max_pixels))
            full_texts.append(
                self.processor.apply_chat_template(
                    build_messages(answer), tokenize=False, add_generation_prompt=False
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    build_messages(None), tokenize=False, add_generation_prompt=True
                )
            )

        batch = self.processor(
            text=full_texts, images=images, return_tensors="pt", padding=True
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask image placeholder tokens — they are inputs, never prediction targets.
        for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            tid = self.processor.tokenizer.convert_tokens_to_ids(token)
            if tid is not None and tid >= 0:
                labels[labels == tid] = -100

        # Mask the prompt span. Length is measured per-sample with the same image,
        # because the processor expands the image placeholder into a variable
        # number of visual tokens depending on resolution.
        for i, prompt in enumerate(prompt_texts):
            n = self.processor(
                text=[prompt], images=[images[i]], return_tensors="pt"
            )["input_ids"].shape[1]
            labels[i, :n] = -100

        batch["labels"] = labels
        return batch


if __name__ == "__main__":
    sample = {"gt_parse": {"total": {"total_price": "25.000"}, "menu": [{"nm": "Kopi"}]}}
    gt = parse_ground_truth(json.dumps(sample))
    print(target_json(gt))
    assert len(build_messages()) == 2 and len(build_messages("{}")) == 3
    print("data self-test passed")
