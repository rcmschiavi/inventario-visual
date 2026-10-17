"""Backends de detecção.

O processador conversa com o modelo por meio do protocolo :class:`Detector`.
Assim os testes usam um detector falso e não precisam de PyTorch/GPU, e trocar
o YOLO-World por outro modelo de vocabulário aberto não afeta o resto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..keyframes import ImageBGR

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RawDetection:
    """Saída bruta do modelo, em pixels da imagem original."""

    class_index: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector(Protocol):
    def detect(self, image: ImageBGR) -> list[RawDetection]: ...

    @property
    def class_names(self) -> list[str]: ...


def resolve_device(device: str) -> str:
    """``"auto"`` vira ``"cuda:0"`` se houver GPU NVIDIA disponível, senão ``"cpu"``."""
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _fp16_kwargs() -> dict[str, Any]:
    """Argumento de FP16 conforme a versão: ``quantize=16`` (novas) ou ``half=True``."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}


class UltralyticsYoloWorld:
    """YOLO-World via pacote ``ultralytics`` (instalado com o extra ``visao``)."""

    def __init__(
        self,
        weights: Path,
        prompts: list[str] | None,
        device: str = "auto",
        conf: float = 0.15,
        iou: float = 0.5,
        imgsz: int = 640,
        max_det: int = 100,
        half: bool = True,
        agnostic_nms: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise ImportError(
                "Pacote 'ultralytics' não instalado. Rode:  uv sync --extra visao"
            ) from exc

        self.device = resolve_device(device)
        weights.parent.mkdir(parents=True, exist_ok=True)
        # Se o arquivo não existir, o ultralytics baixa os pesos para esse caminho.
        self.model: Any = YOLOWorld(str(weights))
        if prompts:
            # Codifica os prompts com o CLIP uma única vez. Na primeira execução
            # o CLIP (~340 MB) é baixado para ~/.cache/clip.
            self.model.set_classes(prompts)
        self.model.to(self.device)
        self._predict_kwargs: dict[str, Any] = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "max_det": max_det,
            "agnostic_nms": agnostic_nms,
            "device": self.device,
            "verbose": False,
        }
        if half and self.device.startswith("cuda"):
            self._predict_kwargs.update(_fp16_kwargs())
        logger.info(
            "YOLO-World carregado (%s, %d classes, dispositivo=%s)",
            weights.name,
            len(self.class_names),
            self.device,
        )

    @property
    def class_names(self) -> list[str]:
        names = self.model.names
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    def warmup(self, size: int = 640) -> None:
        """Primeira inferência é lenta (inicialização CUDA); faz isso no setup."""
        self.detect(np.zeros((size, size, 3), np.uint8))

    def detect(self, image: ImageBGR) -> list[RawDetection]:
        # Ultralytics aceita ndarray BGR diretamente (mesma convenção do OpenCV).
        result = self.model.predict(image, **self._predict_kwargs)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        return [
            RawDetection(int(c), float(p), *(float(v) for v in b))
            for c, p, b in zip(cls, conf, xyxy, strict=True)
        ]
