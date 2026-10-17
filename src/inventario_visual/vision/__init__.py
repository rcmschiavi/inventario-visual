"""Etapa de visão: detecção (YOLO-World) e descrição (VLM) dos itens.

Os modelos reais exigem os extras ``visao`` e ``vlm``
(``uv sync --extra visao --extra vlm``); os módulos em si importam sem
PyTorch, o que mantém os testes leves.
"""

from .classes import Vocabulary
from .detectors import Detector, RawDetection
from .vlm import ItemDescription, VlmProcessor, VlmSettings, parse_response
from .vlm_backend import Describer
from .yolo_world import Detection, YoloWorldProcessor, YoloWorldSettings

__all__ = [
    "Describer",
    "Detection",
    "Detector",
    "ItemDescription",
    "RawDetection",
    "VlmProcessor",
    "VlmSettings",
    "Vocabulary",
    "YoloWorldProcessor",
    "YoloWorldSettings",
    "parse_response",
]
