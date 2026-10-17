"""Etapa de visão: vocabulário, processador YOLO-World (com detector falso) e modelo real."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from inventario_visual.processors import Keyframe, load_processors
from inventario_visual.vision import (
    RawDetection,
    Vocabulary,
    YoloWorldProcessor,
    YoloWorldSettings,
)

from .conftest import cena_texturizada

# --------------------------------------------------------------------------- #
# Vocabulário
# --------------------------------------------------------------------------- #


def test_vocabulario_formato_completo():
    voc = Vocabulary.parse(
        [
            "# comentário",
            "",
            "monitor: computer monitor",
            "cadeira: chair, office chair   # sinônimos",
            "garrafa",
            "assento: chair",  # prompt repetido -> ignorado (primeiro rótulo vence)
        ]
    )
    assert voc.prompts == ("computer monitor", "chair", "office chair", "garrafa")
    assert voc.labels == ("monitor", "cadeira", "cadeira", "garrafa")
    assert voc.unique_labels == ["monitor", "cadeira", "garrafa"]
    assert voc.label_for(2) == "cadeira" and voc.prompt_for(2) == "office chair"


def test_vocabulario_invalido():
    with pytest.raises(ValueError):
        Vocabulary.parse(["# só comentário"])
    with pytest.raises(ValueError):
        Vocabulary.parse([": sem rótulo"])


def test_arquivo_de_classes_do_projeto_eh_valido():
    raiz = Path(__file__).resolve().parent.parent
    voc = Vocabulary.from_file(raiz / "config" / "classes.txt")
    assert len(voc.unique_labels) >= 20
    assert "notebook" in voc.unique_labels


# --------------------------------------------------------------------------- #
# Processador com detector falso
# --------------------------------------------------------------------------- #


class DetectorFalso:
    """Devolve sempre as mesmas caixas, como se fosse o YOLO-World."""

    def __init__(self, raws: list[RawDetection], names: list[str]) -> None:
        self.raws = raws
        self.names = names
        self.chamadas = 0

    @property
    def class_names(self) -> list[str]:
        return self.names

    def detect(self, image: np.ndarray) -> list[RawDetection]:
        self.chamadas += 1
        return self.raws


VOC = Vocabulary.parse(["monitor: computer monitor", "cadeira: chair, office chair"])


def _processador(tmp_path: Path, raws: list[RawDetection], **kw: Any) -> YoloWorldProcessor:
    settings = YoloWorldSettings(_env_file=None, classes_file=None, **kw)
    proc = YoloWorldProcessor(settings, DetectorFalso(raws, list(VOC.prompts)), VOC)
    proc.setup()
    return proc


def _keyframe(tmp_path: Path, eventos: list[dict[str, Any]]) -> Keyframe:
    sessao = tmp_path / "sessao"
    sessao.mkdir(exist_ok=True)
    img = cena_texturizada(1, h=480, w=640)
    path = sessao / "20261017_120000_000000_0001.jpg"
    cv2.imwrite(str(path), img)
    return Keyframe(
        "sessao", 1, datetime(2026, 10, 17, 12, tzinfo=UTC), img, path, 300.0, 1.0, None,
        notify=eventos.append,
    )


def test_processador_grava_deteccoes_recortes_e_anotada(tmp_path: Path):
    raws = [
        RawDetection(0, 0.81, 10, 20, 210, 170),  # monitor
        RawDetection(2, 0.55, 300, 100, 420, 400),  # "office chair" -> cadeira
        RawDetection(1, 0.92, 450, 50, 700, 900),  # "chair", passa da borda -> recortada
        RawDetection(1, 0.30, 50, 50, 50, 90),  # caixa degenerada -> descartada
    ]
    proc = _processador(tmp_path, raws)
    eventos: list[dict[str, Any]] = []
    kf = _keyframe(tmp_path, eventos)
    proc.process(kf)

    dets = kf.extras["detections"]
    # Ordenadas por confiança e com rótulo do catálogo.
    assert [(d.label, d.prompt) for d in dets] == [
        ("cadeira", "chair"),
        ("monitor", "computer monitor"),
        ("cadeira", "office chair"),
    ]
    assert dets[0].bbox == (450, 50, 640, 480)  # limitada ao tamanho da imagem

    sessao = tmp_path / "sessao"
    linha = json.loads((sessao / "detections.jsonl").read_text(encoding="utf-8"))
    assert linha["counts"] == {"cadeira": 2, "monitor": 1}
    assert linha["file"] == "20261017_120000_000000_0001.jpg"
    assert len(linha["detections"]) == 3

    recortes = sorted((sessao / "crops").glob("*.jpg"))
    assert [p.name for p in recortes] == [
        "20261017_120000_000000_0001_01_cadeira.jpg",
        "20261017_120000_000000_0001_02_monitor.jpg",
        "20261017_120000_000000_0001_03_cadeira.jpg",
    ]
    assert linha["detections"][1]["crop_file"] == "crops/20261017_120000_000000_0001_02_monitor.jpg"
    # Recorte do monitor: caixa 200x150 + 5% de margem de cada lado.
    assert cv2.imread(str(recortes[1])).shape[:2] == (164, 220)

    anotada = cv2.imread(str(sessao / "annotated" / "20261017_120000_000000_0001.jpg"))
    assert anotada.shape == (480, 640, 3)

    assert eventos == [{"type": "detections", "index": 1, "counts": {"cadeira": 2, "monitor": 1}}]


def test_processador_sem_deteccoes_e_sem_arquivos_extras(tmp_path: Path):
    proc = _processador(tmp_path, [], save_crops=False, save_annotated=False)
    eventos: list[dict[str, Any]] = []
    kf = _keyframe(tmp_path, eventos)
    proc.process(kf)
    sessao = tmp_path / "sessao"
    assert json.loads((sessao / "detections.jsonl").read_text())["counts"] == {}
    assert not (sessao / "crops").exists() and not (sessao / "annotated").exists()
    assert eventos[0]["counts"] == {}


def test_vocabulario_padrao_do_modelo(tmp_path: Path):
    """Sem arquivo de classes, os nomes vêm do próprio modelo (ex.: COCO)."""
    settings = YoloWorldSettings(_env_file=None, classes_file=None, save_annotated=False)
    detector = DetectorFalso([RawDetection(1, 0.9, 0, 0, 10, 10)], ["person", "bus"])
    proc = YoloWorldProcessor(settings, detector)
    proc.setup()
    assert [d.label for d in proc.detect(cena_texturizada(1))] == ["bus"]


def test_classes_file_vazio_no_env_significa_vocabulario_padrao(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INVENTARIO_YOLO_CLASSES_FILE", "")
    assert YoloWorldSettings(_env_file=None).classes_file is None
    monkeypatch.setenv("INVENTARIO_YOLO_CLASSES_FILE", "outro/classes.txt")
    assert YoloWorldSettings(_env_file=None).classes_file == Path("outro/classes.txt")


def test_carregavel_pelo_env():
    """O processador é plugável pela mesma string usada em INVENTARIO_PROCESSORS."""
    (proc,) = load_processors(["inventario_visual.vision:YoloWorldProcessor"])
    assert isinstance(proc, YoloWorldProcessor)  # setup() (que carrega o modelo) não é chamado aqui


# --------------------------------------------------------------------------- #
# Modelo real (só com `uv sync --extra visao`)
# --------------------------------------------------------------------------- #

tem_ultralytics = importlib.util.find_spec("ultralytics") is not None


@pytest.mark.visao
@pytest.mark.skipif(not tem_ultralytics, reason="extra 'visao' não instalado")
def test_yolo_world_real_em_imagem_de_exemplo(tmp_path: Path):
    """Roda o YOLO-World pequeno na imagem de exemplo do ultralytics (ônibus)."""
    from ultralytics.utils import ASSETS

    raiz = Path(__file__).resolve().parent.parent
    settings = YoloWorldSettings(
        _env_file=None,
        model="yolov8s-worldv2.pt",
        models_dir=raiz / "models",
        classes_file=None,  # vocabulário padrão: não depende do download do CLIP
        save_annotated=False,
        save_crops=False,
    )
    proc = YoloWorldProcessor(settings)
    proc.setup()
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    rotulos = [d.label for d in proc.detect(img)]
    assert "bus" in rotulos
    assert rotulos.count("person") >= 3
