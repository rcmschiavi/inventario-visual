"""Gravação dos frames-chave em disco.

Estrutura gerada::

    data/keyframes/
      20261017_143005_ab12cd/          <- uma pasta por sessão de câmera
        20261017_143012_481233_0001.jpg
        20261017_143015_102344_0002.jpg
        keyframes.jsonl                <- metadados, uma linha por frame
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from .keyframes import ImageBGR


@dataclass(frozen=True, slots=True)
class SavedFrame:
    path: Path
    metadata: dict[str, object]


class KeyframeStore:
    """Salva JPEGs com timestamp no nome e registra metadados em JSONL."""

    def __init__(self, session_dir: Path, jpeg_quality: int = 92) -> None:
        self.session_dir = session_dir
        self.jpeg_quality = int(jpeg_quality)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.session_dir / "keyframes.jsonl"
        self._lock = threading.Lock()

    def save(
        self,
        image: ImageBGR,
        captured_at: datetime,
        index: int,
        extra: dict[str, object] | None = None,
    ) -> SavedFrame:
        stamp = captured_at.strftime("%Y%m%d_%H%M%S_%f")
        path = self.session_dir / f"{stamp}_{index:04d}.jpg"
        ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise OSError(f"Falha ao gravar {path}")
        h, w = image.shape[:2]
        metadata: dict[str, object] = {
            "index": index,
            "file": path.name,
            "captured_at": captured_at.isoformat(),
            "width": w,
            "height": h,
            **(extra or {}),
        }
        with self._lock, self._metadata_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        return SavedFrame(path=path, metadata=metadata)
