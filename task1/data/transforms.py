"""Deterministic pixel-space interventions on common 224x224 RGB inputs."""

import numpy as np
from PIL import Image, ImageOps


def grayscale(image: Image.Image) -> Image.Image:
    return ImageOps.grayscale(image).convert("RGB")


def hue_rotate(image: Image.Image, degrees: float = 45) -> Image.Image:
    # HSV rotation preserves V and S, hence much of the luminance and geometry.
    hsv = np.asarray(image.convert("HSV")).copy()
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.uint16) + round(degrees * 256 / 360)) % 256
    return Image.fromarray(hsv, "HSV").convert("RGB")


def translate(image: Image.Image, dx: int, dy: int) -> Image.Image:
    if not dx and not dy:
        return image.copy()
    a = np.asarray(image)
    p = max(abs(dx), abs(dy))
    padded = np.pad(a, ((p, p), (p, p), (0, 0)), mode="reflect")
    h, w = a.shape[:2]
    # A positive dx moves content right, so the crop starts further left.
    return Image.fromarray(padded[p - dy : p - dy + h, p - dx : p - dx + w])


def patch_shuffle(image: Image.Image, seed: int, image_id: int) -> Image.Image:
    a = np.asarray(image)
    assert a.shape[:2] == (224, 224)
    patches = a.reshape(4, 56, 4, 56, 3).transpose(0, 2, 1, 3, 4).reshape(16, 56, 56, 3)
    rng = np.random.default_rng(np.random.SeedSequence([seed, image_id]))
    order = rng.permutation(16)
    if np.array_equal(order, np.arange(16)):
        order[0], order[1] = order[1], order[0]
    mixed = patches[order].reshape(4, 4, 56, 56, 3).transpose(0, 2, 1, 3, 4)
    return Image.fromarray(mixed.reshape(224, 224, 3))


def variants(image: Image.Image, image_id: int, seed: int, hue_degrees: float):
    yield "clean", image
    yield "grayscale", grayscale(image)
    yield "hue", hue_rotate(image, hue_degrees)
    yield "patch", patch_shuffle(image, seed, image_id)
    for delta in (8, 16, 32):
        for direction, dx, dy in (("right", delta, 0), ("left", -delta, 0),
                                  ("down", 0, delta), ("up", 0, -delta)):
            yield f"translate_{delta}_{direction}", translate(image, dx, dy)
