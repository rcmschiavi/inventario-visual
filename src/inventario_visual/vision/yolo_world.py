"""Processador de frames-chave com YOLO-World (detecção de vocabulário aberto).

Para cada frame-chave:

1. detecta os objetos do vocabulário (``config/classes.txt``);
2. converte os prompts de volta para os rótulos em português;
3. grava os resultados na pasta da sessão::

       data/keyframes/<sessão>/
         detections.jsonl          # uma linha por frame-chave (contagens + caixas)
         crops/<frame>_01_monitor.jpg
         annotated/<frame>.jpg     # frame com as caixas desenhadas

4. deixa as detecções em ``keyframe.extras["detections"]`` para o próximo
   processador (o modelo de visão-linguagem) e avisa o celular.

Ativação no ``.env``::

    INVENTARIO_PROCESSORS=inventario_visual.vision:YoloWorldProcessor
"""

from __future__ import annotations

import json
import logging
import threading
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..config import Settings
from ..keyframes import ImageBGR
from ..processors import FrameProcessor, Keyframe
from .classes import Vocabulary
from .detectors import Detector, RawDetection, UltralyticsYoloWorld

logger = logging.getLogger(__name__)


class YoloWorldSettings(BaseSettings):
    """Parâmetros do YOLO-World (variáveis ``INVENTARIO_YOLO_*`` ou ``.env``)."""

    model_config = SettingsConfigDict(
        env_prefix="INVENTARIO_YOLO_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Pesos: yolov8s/m/l/x-worldv2.pt (maior = mais preciso e mais lento).
    model: str = "yolov8l-worldv2.pt"
    models_dir: Path = Path("models")
    # Vazio = usa as 80 classes padrão (COCO, em inglês) embutidas nos pesos.
    classes_file: Path | None = Path("config/classes.txt")
    device: str = "auto"  # "auto", "cpu", "cuda:0"...
    conf: float = Field(default=0.15, gt=0, lt=1)
    iou: float = Field(default=0.5, gt=0, lt=1)
    # Resolução de inferência. 1280 ajuda com objetos pequenos (mais lento).
    imgsz: int = Field(default=640, ge=320)
    max_det: int = Field(default=100, ge=1)
    half: bool = True  # FP16 na GPU
    # Suprime caixas sobrepostas mesmo de classes diferentes (útil com sinônimos).
    agnostic_nms: bool = True
    save_crops: bool = True
    save_annotated: bool = True
    # Margem extra ao recortar cada objeto (fração do tamanho da caixa).
    crop_padding: float = Field(default=0.05, ge=0, le=1)

    @field_validator("classes_file", mode="before")
    @classmethod
    def _vazio_eh_none(cls, v: object) -> object:
        # `INVENTARIO_YOLO_CLASSES_FILE=` (vazio) significa "sem arquivo", não Path(".").
        return None if v is None or (isinstance(v, str) and not v.strip()) else v

    @property
    def weights_path(self) -> Path:
        return self.models_dir / self.model


@dataclass(frozen=True, slots=True)
class Detection:
    """Objeto detectado, já com o rótulo do catálogo."""

    label: str
    prompt: str
    confidence: float
    # Caixa em pixels da imagem original: (x1, y1, x2, y2), inteiros.
    bbox: tuple[int, int, int, int]
    crop_file: str | None = None


def _ascii(text: str) -> str:
    """cv2.putText não desenha acentos; remove-os só para a imagem anotada."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _color(label: str) -> tuple[int, int, int]:
    """Cor estável por rótulo (o mesmo item sempre com a mesma cor)."""
    h = sum(ord(c) * 31**i for i, c in enumerate(label)) % 180
    bgr = cv2.cvtColor(np.array([[[h, 200, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def summarize(detections: list[Detection]) -> dict[str, int]:
    """Contagem por rótulo, do mais frequente para o menos."""
    return dict(Counter(d.label for d in detections).most_common())


class YoloWorldProcessor(FrameProcessor):
    """Detecta objetos do vocabulário em cada frame-chave."""

    name = "yolo-world"

    def __init__(
        self,
        settings: YoloWorldSettings | None = None,
        detector: Detector | None = None,
        vocabulary: Vocabulary | None = None,
    ) -> None:
        self.settings = settings or YoloWorldSettings()
        self.detector = detector
        self.vocabulary = vocabulary
        self._output_root = Settings().output_dir
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        s = self.settings
        if self.vocabulary is None and s.classes_file is not None:
            self.vocabulary = Vocabulary.from_file(s.classes_file)
            logger.info(
                "Vocabulário: %d rótulos, %d prompts (%s)",
                len(self.vocabulary.unique_labels),
                len(self.vocabulary.prompts),
                s.classes_file,
            )
        if self.detector is None:
            detector = UltralyticsYoloWorld(
                s.weights_path,
                list(self.vocabulary.prompts) if self.vocabulary else None,
                device=s.device,
                conf=s.conf,
                iou=s.iou,
                imgsz=s.imgsz,
                max_det=s.max_det,
                half=s.half,
                agnostic_nms=s.agnostic_nms,
            )
            detector.warmup(s.imgsz)
            self.detector = detector

    # ---------------------------------------------------------------- process

    def detect(self, image: ImageBGR) -> list[Detection]:
        """Roda o detector e converte para :class:`Detection` (sem gravar nada)."""
        if self.detector is None:
            raise RuntimeError("setup() não foi chamado")
        h, w = image.shape[:2]
        out: list[Detection] = []
        for raw in sorted(self.detector.detect(image), key=lambda r: -r.confidence):
            label, prompt = self._names(raw)
            x1, y1 = max(0, int(round(raw.x1))), max(0, int(round(raw.y1)))
            x2, y2 = min(w, int(round(raw.x2))), min(h, int(round(raw.y2)))
            if x2 <= x1 or y2 <= y1:
                continue  # caixa degenerada
            out.append(Detection(label, prompt, round(raw.confidence, 4), (x1, y1, x2, y2)))
        return out

    def process(self, keyframe: Keyframe) -> None:
        detections = self.detect(keyframe.image)
        session_dir, stem = self._location(keyframe)

        if self.settings.save_crops and detections:
            detections = self._save_crops(keyframe.image, detections, session_dir / "crops", stem)
        if self.settings.save_annotated:
            self._save_annotated(keyframe.image, detections, session_dir / "annotated" / f"{stem}.jpg")

        counts = summarize(detections)
        record = {
            "index": keyframe.index,
            "file": keyframe.path.name if keyframe.path else None,
            "captured_at": keyframe.captured_at.isoformat(),
            "counts": counts,
            "detections": [asdict(d) for d in detections],
        }
        session_dir.mkdir(parents=True, exist_ok=True)
        with self._lock, (session_dir / "detections.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        keyframe.extras["detections"] = detections
        keyframe.send_event({"type": "detections", "index": keyframe.index, "counts": counts})
        logger.info(
            "[%s] frame #%d: %s",
            keyframe.session_id,
            keyframe.index,
            ", ".join(f"{n} {lbl}" for lbl, n in counts.items()) or "nada detectado",
        )

    # --------------------------------------------------------------- auxiliares

    def _names(self, raw: RawDetection) -> tuple[str, str]:
        if self.vocabulary is not None:
            return self.vocabulary.label_for(raw.class_index), self.vocabulary.prompt_for(raw.class_index)
        assert self.detector is not None
        name = self.detector.class_names[raw.class_index]
        return name, name

    def _location(self, keyframe: Keyframe) -> tuple[Path, str]:
        if keyframe.path is not None:
            return keyframe.path.parent, keyframe.path.stem
        # O frame não foi salvo: grava as detecções mesmo assim na pasta da sessão.
        return self._output_root / keyframe.session_id, f"frame_{keyframe.index:04d}"

    def _save_crops(
        self, image: ImageBGR, detections: list[Detection], crops_dir: Path, stem: str
    ) -> list[Detection]:
        crops_dir.mkdir(parents=True, exist_ok=True)
        h, w = image.shape[:2]
        pad = self.settings.crop_padding
        saved: list[Detection] = []
        for i, d in enumerate(detections, start=1):
            x1, y1, x2, y2 = d.bbox
            px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
            crop = image[max(0, y1 - py) : min(h, y2 + py), max(0, x1 - px) : min(w, x2 + px)]
            name = f"{stem}_{i:02d}_{_ascii(d.label).replace(' ', '-')}.jpg"
            cv2.imwrite(str(crops_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved.append(Detection(d.label, d.prompt, d.confidence, d.bbox, f"crops/{name}"))
        return saved

    def _save_annotated(self, image: ImageBGR, detections: list[Detection], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas = image.copy()
        thickness = max(2, round(max(image.shape[:2]) / 600))
        scale = thickness * 0.35
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = _color(d.label)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            text = f"{_ascii(d.label)} {d.confidence:.2f}"
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            ty = max(th + base, y1)
            cv2.rectangle(canvas, (x1, ty - th - base), (x1 + tw, ty), color, -1)
            cv2.putText(
                canvas, text, (x1, ty - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness
            )
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
