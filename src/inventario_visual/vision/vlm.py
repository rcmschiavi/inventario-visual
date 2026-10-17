"""Descrição dos itens detectados com um modelo de visão-linguagem (VLM).

Para cada objeto que o YOLO-World encontrou, o VLM recebe o recorte e devolve
uma ficha em JSON: nome, categoria, marca, modelo, cor, estado, texto visível
e se o recorte realmente corresponde ao rótulo detectado (filtra falsos
positivos do detector).

Dois modos de uso:

* **Ao vivo** — processador logo depois do YOLO-World::

      INVENTARIO_PROCESSORS=inventario_visual.vision:YoloWorldProcessor,inventario_visual.vision:VlmProcessor

* **Em lote** — depois de filmar, descreve uma sessão inteira (retoma de onde
  parou, então pode ser interrompido)::

      inventario-descrever data/keyframes/<sessão>

Saídas na pasta da sessão: ``descriptions.jsonl`` (completo) e
``inventario.csv`` (uma linha por item, abre no LibreOffice/Excel).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..keyframes import ImageBGR
from ..processors import FrameProcessor, Keyframe
from .vlm_backend import Describer, ImageRGB, Quantization, TransformersVlm
from .yolo_world import Detection

logger = logging.getLogger(__name__)


class VlmSettings(BaseSettings):
    """Parâmetros do VLM (variáveis ``INVENTARIO_VLM_*`` ou ``.env``)."""

    model_config = SettingsConfigDict(
        env_prefix="INVENTARIO_VLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Qualquer modelo do Hugging Face compatível com AutoModelForImageTextToText.
    model: str = "Qwen/Qwen3-VL-4B-Instruct"
    quantization: Quantization = "4bit"
    device: str = "auto"
    # Onde guardar os pesos (vazio = cache padrão do Hugging Face, ~/.cache/huggingface).
    cache_dir: Path | None = None
    # Lado maior do recorte enviado ao modelo: mais pixels = mais detalhe e mais VRAM.
    max_side: int = Field(default=768, ge=64)
    # Recortes menores que isso (em pixels, no lado menor) são ignorados.
    min_crop_px: int = Field(default=40, ge=1)
    crop_padding: float = Field(default=0.08, ge=0, le=1)
    # Máximo de itens descritos por frame-chave (os de maior confiança primeiro).
    max_items: int = Field(default=6, ge=1)
    max_new_tokens: int = Field(default=200, ge=16)
    # Arquivo com um prompt alternativo; use {rotulo} onde entra o rótulo detectado.
    prompt_file: Path | None = None

    @field_validator("cache_dir", "prompt_file", mode="before")
    @classmethod
    def _vazio_eh_none(cls, v: object) -> object:
        return None if v is None or (isinstance(v, str) and not v.strip()) else v


# --------------------------------------------------------------------------- #
# Prompt e interpretação da resposta
# --------------------------------------------------------------------------- #

DEFAULT_PROMPT = """\
Você está catalogando itens para um inventário. A imagem é o recorte de um objeto \
que um detector automático identificou como "{rotulo}".

Responda SOMENTE com um objeto JSON, sem texto antes ou depois, com estas chaves:
- "confere": true se a imagem realmente mostra um(a) {rotulo}; false caso contrário
- "nome": nome curto do item, em português
- "categoria": categoria geral (ex.: informática, mobiliário, ferramenta, eletrodoméstico)
- "marca": marca, só se estiver legível ou for inconfundível; senão null
- "modelo": modelo, só se estiver legível; senão null
- "cor": cor predominante
- "estado": "novo", "bom", "usado" ou "danificado"; null se não der para avaliar
- "texto_visivel": textos legíveis no objeto (rótulos, etiquetas, números de série); senão null
- "descricao": uma frase descrevendo o item

Não invente marca, modelo ou texto: na dúvida, use null."""


@dataclass(slots=True)
class ItemDescription:
    """Ficha de um item, como devolvida (e normalizada) do VLM."""

    confere: bool | None = None
    nome: str | None = None
    categoria: str | None = None
    marca: str | None = None
    modelo: str | None = None
    cor: str | None = None
    estado: str | None = None
    texto_visivel: str | None = None
    descricao: str | None = None
    # Preenchido quando a resposta não pôde ser interpretada como JSON.
    erro: str | None = None


_NULOS = {"", "null", "none", "n/a", "na", "desconhecido", "desconhecida", "não identificado", "-"}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _texto(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, list):
        v = "; ".join(str(x) for x in v if x is not None)
    s = str(v).strip()
    return None if s.lower() in _NULOS else s


def _bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "sim", "yes", "1"}:
        return True
    if s in {"false", "não", "nao", "no", "0"}:
        return False
    return None


def parse_response(text: str) -> ItemDescription:
    """Extrai a ficha do texto do modelo (tolera ```json``` e texto em volta)."""
    match = _JSON_RE.search(text)
    if match is None:
        return ItemDescription(erro="resposta sem JSON", descricao=_texto(text))
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return ItemDescription(erro=f"JSON inválido: {exc.msg}", descricao=_texto(text))
    if not isinstance(data, dict):
        return ItemDescription(erro="JSON não é um objeto")
    data = {str(k).strip().lower(): v for k, v in data.items()}
    item = ItemDescription(confere=_bool(data.get("confere")))
    for f in fields(ItemDescription):
        if f.name not in {"confere", "erro"}:
            setattr(item, f.name, _texto(data.get(f.name)))
    return item


# --------------------------------------------------------------------------- #
# Recortes
# --------------------------------------------------------------------------- #


def prepare_crop(
    image: ImageBGR, bbox: tuple[int, int, int, int], padding: float, max_side: int
) -> ImageRGB:
    """Recorta com margem, limita o lado maior e converte para RGB."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = int((x2 - x1) * padding), int((y2 - y1) * padding)
    crop = image[max(0, y1 - py) : min(h, y2 + py), max(0, x1 - px) : min(w, x2 + px)]
    ch, cw = crop.shape[:2]
    if max(ch, cw) > max_side:
        scale = max_side / max(ch, cw)
        size = (max(1, round(cw * scale)), max(1, round(ch * scale)))
        crop = np.asarray(cv2.resize(crop, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def select_detections(detections: Iterable[Detection], settings: VlmSettings) -> list[Detection]:
    """Filtra recortes pequenos demais e limita a quantidade (maior confiança primeiro)."""
    ok = [
        d
        for d in sorted(detections, key=lambda d: -d.confidence)
        if min(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]) >= settings.min_crop_px
    ]
    return ok[: settings.max_items]


# --------------------------------------------------------------------------- #
# Gravação
# --------------------------------------------------------------------------- #

CSV_COLUMNS = [
    "frame",
    "indice_frame",
    "capturado_em",
    "rotulo_detector",
    "confianca_detector",
    *[f.name for f in fields(ItemDescription)],
    "recorte",
    "bbox",
]


class DescriptionWriter:
    """Acrescenta resultados em ``descriptions.jsonl`` e ``inventario.csv``."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        item = record["item"]
        row = {
            "frame": record["file"],
            "indice_frame": record["index"],
            "capturado_em": record.get("captured_at"),
            "rotulo_detector": record["label"],
            "confianca_detector": record["detection_confidence"],
            **{k: ("" if v is None else v) for k, v in item.items()},
            "recorte": record.get("crop_file") or "",
            "bbox": " ".join(str(v) for v in record["bbox"]),
        }
        csv_path = self.session_dir / "inventario.csv"
        with self._lock:
            with (self.session_dir / "descriptions.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            new = not csv_path.exists()
            # utf-8-sig: o Excel reconhece os acentos corretamente.
            with csv_path.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
                if new:
                    writer.writeheader()
                writer.writerow(row)


def _record_key(file: str | None, label: str, bbox: Iterable[int]) -> str:
    return f"{file}|{label}|{','.join(str(int(v)) for v in bbox)}"


def done_keys(session_dir: Path) -> set[str]:
    """Itens já descritos (para retomar um processamento em lote)."""
    path = session_dir / "descriptions.jsonl"
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add(_record_key(r["file"], r["label"], r["bbox"]))
    return keys


# --------------------------------------------------------------------------- #
# Núcleo compartilhado (ao vivo e em lote)
# --------------------------------------------------------------------------- #


class ItemDescriber:
    """Descreve detecções de um frame e grava os resultados."""

    def __init__(self, settings: VlmSettings, describer: Describer | None = None) -> None:
        self.settings = settings
        self.describer = describer
        self.prompt_template = (
            settings.prompt_file.read_text(encoding="utf-8") if settings.prompt_file else DEFAULT_PROMPT
        )

    def load(self) -> None:
        if self.describer is None:
            s = self.settings
            self.describer = TransformersVlm(s.model, s.quantization, s.device, s.max_new_tokens, s.cache_dir)

    def describe_frame(
        self,
        image: ImageBGR,
        detections: Iterable[Detection],
        frame: dict[str, Any],
        writer: DescriptionWriter,
        skip: set[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Gera (e grava) um registro por detecção selecionada.

        ``frame`` traz ``file``, ``index`` e ``captured_at`` do frame-chave.
        """
        if self.describer is None:
            raise RuntimeError("load() não foi chamado")
        for det in select_detections(detections, self.settings):
            key = _record_key(frame["file"], det.label, det.bbox)
            if skip is not None and key in skip:
                continue
            crop = prepare_crop(image, det.bbox, self.settings.crop_padding, self.settings.max_side)
            t0 = time.perf_counter()
            try:
                raw = self.describer.describe(crop, self.prompt_template.replace("{rotulo}", det.label))
                item = parse_response(raw)
            except Exception as exc:  # erro do modelo não derruba o lote
                logger.exception("Falha ao descrever %s", det.label)
                raw, item = "", ItemDescription(erro=f"{type(exc).__name__}: {exc}")
            record = {
                **frame,
                "label": det.label,
                "detection_confidence": det.confidence,
                "bbox": list(det.bbox),
                "crop_file": det.crop_file,
                "item": asdict(item),
                "raw": raw,
                "vlm_model": self.settings.model,
                "seconds": round(time.perf_counter() - t0, 2),
            }
            writer.write(record)
            yield record


def _display_name(record: dict[str, Any]) -> str:
    item = record["item"]
    nome = item.get("nome") or record["label"]
    if item.get("marca"):
        nome += f" {item['marca']}"
    return f"{nome} (?)" if item.get("confere") is False else nome


# --------------------------------------------------------------------------- #
# Modo ao vivo
# --------------------------------------------------------------------------- #


class VlmProcessor(FrameProcessor):
    """Descreve, ao vivo, os itens que o YOLO-World detectou em cada frame-chave.

    Deve vir **depois** do ``YoloWorldProcessor`` em ``INVENTARIO_PROCESSORS``.
    """

    name = "vlm"

    def __init__(self, settings: VlmSettings | None = None, describer: Describer | None = None) -> None:
        self.core = ItemDescriber(settings or VlmSettings(), describer)
        self._writers: dict[Path, DescriptionWriter] = {}
        self._warned = False

    def setup(self) -> None:
        self.core.load()

    def process(self, keyframe: Keyframe) -> None:
        detections: list[Detection] | None = keyframe.extras.get("detections")
        if detections is None:
            if not self._warned:
                logger.warning("VlmProcessor sem detecções: coloque o YoloWorldProcessor antes dele")
                self._warned = True
            return
        if keyframe.path is None:
            return  # sem o JPEG em disco não há como referenciar o item depois
        session_dir = keyframe.path.parent
        writer = self._writers.setdefault(session_dir, DescriptionWriter(session_dir))
        frame = {
            "file": keyframe.path.name,
            "index": keyframe.index,
            "captured_at": keyframe.captured_at.isoformat(),
        }
        records = list(self.core.describe_frame(keyframe.image, detections, frame, writer))
        keyframe.extras["descriptions"] = records
        if records:
            nomes = [_display_name(r) for r in records]
            keyframe.send_event({"type": "descriptions", "index": keyframe.index, "itens": nomes})
            logger.info("[%s] frame #%d: %s", keyframe.session_id, keyframe.index, "; ".join(nomes))


# --------------------------------------------------------------------------- #
# Modo em lote: inventario-descrever <sessão>
# --------------------------------------------------------------------------- #


def describe_session(session_dir: Path, core: ItemDescriber, refazer: bool = False) -> int:
    """Descreve todas as detecções de uma sessão; retorna quantos itens foram descritos."""
    det_path = session_dir / "detections.jsonl"
    if not det_path.exists():
        raise FileNotFoundError(f"{det_path} não existe: rode o YOLO-World antes")
    if refazer:
        for name in ("descriptions.jsonl", "inventario.csv"):
            (session_dir / name).unlink(missing_ok=True)
    skip = done_keys(session_dir)
    writer = DescriptionWriter(session_dir)
    total = 0
    for line in det_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("file") or not r["detections"]:
            continue
        image = cv2.imread(str(session_dir / r["file"]))
        if image is None:
            logger.warning("Frame %s não encontrado; pulando", r["file"])
            continue
        detections = [
            Detection(d["label"], d["prompt"], d["confidence"], tuple(d["bbox"]), d.get("crop_file"))
            for d in r["detections"]
        ]
        frame = {"file": r["file"], "index": r["index"], "captured_at": r.get("captured_at")}
        for rec in core.describe_frame(np.asarray(image, dtype=np.uint8), detections, frame, writer, skip):
            total += 1
            print(f"  #{rec['index']:<4} {_display_name(rec):<40} ({rec['seconds']:.1f}s)")
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Descreve com o VLM os itens detectados de uma sessão")
    parser.add_argument("sessoes", type=Path, nargs="+", help="pasta(s) data/keyframes/<sessão>")
    parser.add_argument("--refazer", action="store_true", help="apaga as descrições e começa de novo")
    parser.add_argument("--max-itens", type=int, help="sobrescreve INVENTARIO_VLM_MAX_ITEMS")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    settings = VlmSettings()
    if args.max_itens:
        settings.max_items = args.max_itens
    core = ItemDescriber(settings)
    core.load()
    for sessao in args.sessoes:
        print(f"Sessão {sessao}:")
        n = describe_session(sessao, core, args.refazer)
        print(f"  {n} item(ns) descrito(s) -> {sessao / 'inventario.csv'}")


if __name__ == "__main__":
    main()
