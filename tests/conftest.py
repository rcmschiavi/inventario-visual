"""Geradores de imagens sintéticas para os testes."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

Image = NDArray[np.uint8]


def cena_texturizada(seed: int, h: int = 240, w: int = 320) -> Image:
    """Cena 'nítida': blocos aleatórios + retângulos com bordas fortes."""
    rng = np.random.default_rng(seed)
    # Blocos 8x8 de cores aleatórias ampliados sem interpolação -> bordas duras.
    blocos = rng.integers(0, 256, size=(h // 8, w // 8, 3), dtype=np.uint8)
    img = cv2.resize(blocos, (w, h), interpolation=cv2.INTER_NEAREST)
    for _ in range(6):
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        cor = tuple(int(c) for c in rng.integers(0, 256, 3))
        cv2.rectangle(img, (x, y), (x + 50, y + 40), cor, thickness=-1)
    return img


def borrar(img: Image, k: int = 21) -> Image:
    return cv2.GaussianBlur(img, (k, k), 0)


def deslocar(img: Image, dx: int, dy: int = 0) -> Image:
    """Simula a câmera andando (translação)."""
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def ruido_sensor(img: Image, seed: int, sigma: float = 2.0) -> Image:
    """Ruído leve, como o de um sensor real com a câmera parada."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


@pytest.fixture
def cena() -> Image:
    return cena_texturizada(1)
