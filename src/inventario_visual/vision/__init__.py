"""Etapa de visão: detecção de objetos nos frames-chave.

Requer o extra ``visao`` (``uv sync --extra visao``) para o YOLO-World real;
os módulos em si importam sem PyTorch, o que mantém os testes leves.
"""

from .classes import Vocabulary
from .detectors import Detector, RawDetection
from .yolo_world import Detection, YoloWorldProcessor, YoloWorldSettings

__all__ = [
    "Detection",
    "Detector",
    "RawDetection",
    "Vocabulary",
    "YoloWorldProcessor",
    "YoloWorldSettings",
]
