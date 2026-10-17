"""Seletor aplicado a um arquivo de vídeo sintético gravado com OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from inventario_visual.keyframes import KeyframeSelectorConfig
from inventario_visual.offline import process_video

from .conftest import cena_texturizada, deslocar


@pytest.fixture
def video_sintetico(tmp_path: Path) -> Path:
    """Vídeo MJPG de 4 s a 30 fps: A parado (1 s), pan (1 s), B parado (2 s)."""
    path = tmp_path / "sintetico.avi"
    a, b = cena_texturizada(1), cena_texturizada(2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("OpenCV sem suporte a VideoWriter MJPG neste ambiente")
    frames: list[np.ndarray] = [a] * 30 + [deslocar(a, 10 * i) for i in range(1, 31)] + [b] * 60
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def test_processa_video_e_salva(video_sintetico: Path, tmp_path: Path):
    cfg = KeyframeSelectorConfig(sample_fps=5, analysis_width=320, min_keyframe_interval_s=0.5)
    saida = tmp_path / "saida"
    tempos = process_video(video_sintetico, cfg, saida)
    assert len(tempos) == 2
    assert tempos[0] < 1.0 and tempos[1] > 2.0
    jpgs = sorted(saida.glob("*.jpg"))
    assert len(jpgs) == 2
    assert (saida / "keyframes.jsonl").read_text(encoding="utf-8").count("\n") == 2
