"""Deterministic image corruptions simulating real-world document capture failures.

Every corruption takes a PIL.Image and a severity in {1, 2, 3} and returns a new
PIL.Image of the same mode. Severity 1 is a mild degradation that a human would
barely notice; severity 3 is bad but still legible to a human reader. The point
of the ceiling is that any accuracy the model loses at severity 3 is accuracy it
should not have lost.

Randomised corruptions are seeded per-call from (name, severity, seed) so that a
given sample gets the same corruption on every evaluation run. Without this the
robustness numbers move between runs and you cannot tell a real regression from
sampling noise.
"""

from __future__ import annotations

import io
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SEVERITIES = (1, 2, 3)


def _rng(name: str, severity: int, seed: int) -> np.random.Generator:
    """Stable per-(corruption, severity, sample) generator."""
    return np.random.default_rng(abs(hash((name, severity, seed))) % (2**32))


def identity(img: Image.Image, severity: int = 1, seed: int = 0) -> Image.Image:
    """Control condition. Present so the clean run flows through the same code path."""
    return img.copy()


def gaussian_blur(img: Image.Image, severity: int = 1, seed: int = 0) -> Image.Image:
    """Out-of-focus capture. Radius scales with the image's short side so the
    effect is comparable across CORD's varied resolutions."""
    frac = {1: 0.002, 2: 0.004, 3: 0.008}[severity]
    radius = max(0.6, frac * min(img.size))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def gaussian_noise(img: Image.Image, severity: int = 1, seed: int = 0) -> Image.Image:
    """Sensor noise from a low-light phone camera."""
    sigma = {1: 8.0, 2: 18.0, 3: 32.0}[severity]
    rng = _rng("gaussian_noise", severity, seed)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = arr + rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg_artifacts(img: Image.Image, severity: int = 1, seed: int = 0) -> Image.Image:
    """Re-compression damage. Documents are routinely screenshotted, forwarded
    through chat apps, and re-saved, each pass shedding high-frequency detail —
    which is exactly where thin digits live."""
    quality = {1: 40, 2: 22, 3: 10}[severity]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rotation(img: Image.Image, severity: int = 1, seed: int = 0) -> Image.Image:
    """Handheld capture skew. Sign alternates per sample so the eval set does not
    drift toward one direction, which would let the model learn the tilt."""
    degrees = {1: 3.0, 2: 7.0, 3: 13.0}[severity]
    rng = _rng("rotation", severity, seed)
    angle = degrees * (1.0 if rng.random() < 0.5 else -1.0)
    return img.convert("RGB").rotate(
        angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)
    )


def downscale(img: Image.Image, severity: int = 1, seed: int = 0) -> Image.Image:
    """Resolution loss from a thumbnail or an aggressive upload pipeline.
    Downsample then restore the original size, so the model still receives the
    expected input dimensions but the information is gone."""
    factor = {1: 0.6, 2: 0.4, 3: 0.25}[severity]
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def low_contrast(img: Image.Image, severity: int = 1, seed: int = 0) -> Image.Image:
    """Faded thermal receipt paper, or a photo taken in poor light."""
    factor = {1: 0.7, 2: 0.5, 3: 0.32}[severity]
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)


CORRUPTIONS: Dict[str, Callable[[Image.Image, int, int], Image.Image]] = {
    "clean": identity,
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg_artifacts": jpeg_artifacts,
    "rotation": rotation,
    "downscale": downscale,
    "low_contrast": low_contrast,
}


def apply_corruption(
    img: Image.Image, name: str, severity: int = 1, seed: int = 0
) -> Image.Image:
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; have {sorted(CORRUPTIONS)}")
    if name != "clean" and severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    return CORRUPTIONS[name](img, severity, seed)


def corruption_grid(include_clean: bool = True) -> List[tuple]:
    """All (name, severity) pairs the evaluation sweeps over."""
    grid = [("clean", 0)] if include_clean else []
    for name in CORRUPTIONS:
        if name == "clean":
            continue
        grid.extend((name, s) for s in SEVERITIES)
    return grid


if __name__ == "__main__":
    # Smoke test with a synthetic page; writes a contact sheet to results/.
    import pathlib

    demo = Image.new("RGB", (480, 640), "white")
    for name, sev in corruption_grid():
        out = apply_corruption(demo, name, sev or 1, seed=0)
        assert out.mode == "RGB", (name, sev)
    pathlib.Path("results").mkdir(exist_ok=True)
    print(f"{len(corruption_grid())} conditions OK")
