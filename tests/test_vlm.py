"""Etapa 2: descrição dos itens com VLM (backend falso, sem PyTorch)."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from inventario_visual.processors import Keyframe
from inventario_visual.vision import Detection, VlmProcessor, VlmSettings, parse_response
from inventario_visual.vision.vlm import ItemDescriber, describe_session, prepare_crop, select_detections

from .conftest import cena_texturizada

# --------------------------------------------------------------------------- #
# Interpretação da resposta
# --------------------------------------------------------------------------- #


def test_resposta_com_cerca_de_codigo_e_texto_em_volta():
    texto = """Claro! Segue:
```json
{"confere": true, "nome": "Furadeira de impacto", "categoria": "ferramenta",
 "marca": "Bosch", "modelo": "GSB 13 RE", "cor": "azul", "estado": "bom",
 "texto_visivel": ["BOSCH", "GSB 13 RE"], "descricao": "Furadeira azul com mandril."}
```"""
    item = parse_response(texto)
    assert item.erro is None
    assert item.confere is True
    assert (item.nome, item.marca, item.modelo) == ("Furadeira de impacto", "Bosch", "GSB 13 RE")
    assert item.texto_visivel == "BOSCH; GSB 13 RE"  # lista vira texto


def test_normaliza_nulos_e_booleanos_em_texto():
    item = parse_response('{"Confere": "não", "nome": "Cadeira", "marca": "null", "modelo": "", '
                          '"cor": "desconhecida", "estado": null}')
    assert item.confere is False
    assert item.nome == "Cadeira"
    assert item.marca is None and item.modelo is None and item.cor is None and item.estado is None


@pytest.mark.parametrize(
    ("texto", "erro"),
    [("Não sei o que é isso.", "resposta sem JSON"), ('{"nome": "x",}', "JSON inválido")],
)
def test_resposta_invalida_vira_erro_sem_excecao(texto: str, erro: str):
    item = parse_response(texto)
    assert item.erro is not None and item.erro.startswith(erro)


# --------------------------------------------------------------------------- #
# Recortes e seleção
# --------------------------------------------------------------------------- #


def test_recorte_com_margem_reducao_e_rgb():
    img = np.zeros((1000, 1000, 3), np.uint8)
    img[:, :] = (255, 0, 0)  # azul em BGR
    crop = prepare_crop(img, (100, 100, 300, 200), padding=0.1, max_side=1000)
    assert crop.shape == (120, 240, 3)  # 200x100 + 10% de cada lado
    assert tuple(crop[0, 0]) == (0, 0, 255)  # virou RGB
    grande = prepare_crop(img, (0, 0, 1000, 500), padding=0.0, max_side=400)
    assert grande.shape == (200, 400, 3)


def _det(label: str, conf: float, bbox: tuple[int, int, int, int]) -> Detection:
    return Detection(label, label, conf, bbox)


def test_selecao_filtra_pequenos_e_limita_quantidade():
    s = VlmSettings(_env_file=None, min_crop_px=40, max_items=2)
    dets = [
        _det("a", 0.5, (0, 0, 100, 100)),
        _det("pequeno", 0.99, (0, 0, 30, 300)),  # lado menor 30 < 40
        _det("b", 0.9, (0, 0, 50, 50)),
        _det("c", 0.7, (0, 0, 80, 80)),
    ]
    assert [d.label for d in select_detections(dets, s)] == ["b", "c"]


def test_campos_vazios_no_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INVENTARIO_VLM_PROMPT_FILE", "")
    monkeypatch.setenv("INVENTARIO_VLM_CACHE_DIR", " ")
    s = VlmSettings(_env_file=None)
    assert s.prompt_file is None and s.cache_dir is None
    assert s.model == "Qwen/Qwen3-VL-4B-Instruct" and s.quantization == "4bit"


# --------------------------------------------------------------------------- #
# Processador ao vivo e modo em lote
# --------------------------------------------------------------------------- #


class VlmFalso:
    """Responde uma ficha fixa conforme o rótulo que aparece no prompt."""

    def __init__(self, falhar_em: str | None = None) -> None:
        self.chamadas: list[tuple[tuple[int, ...], str]] = []
        self.falhar_em = falhar_em

    def describe(self, image: np.ndarray, prompt: str) -> str:
        self.chamadas.append((image.shape, prompt))
        if self.falhar_em and f'"{self.falhar_em}"' in prompt:
            raise RuntimeError("CUDA out of memory")
        if '"monitor"' in prompt:
            return '{"confere": true, "nome": "Monitor", "marca": "Dell", "texto_visivel": "P2419H"}'
        return '```json\n{"confere": false, "nome": "Banqueta", "categoria": "mobiliário"}\n```'


DETS = [
    Detection("monitor", "computer monitor", 0.81, (10, 20, 210, 170), "crops/f_01_monitor.jpg"),
    Detection("cadeira", "chair", 0.55, (300, 100, 420, 400), None),
]


def _sessao(tmp_path: Path) -> tuple[Path, np.ndarray]:
    sessao = tmp_path / "sessao"
    sessao.mkdir()
    img = cena_texturizada(1, h=480, w=640)
    cv2.imwrite(str(sessao / "f.jpg"), img)
    return sessao, img


def test_processador_ao_vivo(tmp_path: Path):
    sessao, img = _sessao(tmp_path)
    vlm = VlmFalso()
    proc = VlmProcessor(VlmSettings(_env_file=None), vlm)
    proc.setup()
    eventos: list[dict[str, Any]] = []
    kf = Keyframe("s", 3, datetime(2026, 10, 17, tzinfo=UTC), img, sessao / "f.jpg", 200.0, 1.0, None,
                  extras={"detections": DETS}, notify=eventos.append)
    proc.process(kf)

    assert [p.count('"monitor"') > 0 for _, p in vlm.chamadas] == [True, False]
    assert all("{rotulo}" not in p for _, p in vlm.chamadas)
    # 200x150 + 8% de margem (16 px e 12 px); à esquerda a margem é cortada na borda (x1=10).
    assert vlm.chamadas[0][0] == (174, 226, 3)

    linhas = (sessao / "descriptions.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(linhas[0])
    assert rec["item"]["marca"] == "Dell" and rec["file"] == "f.jpg" and rec["index"] == 3
    assert rec["crop_file"] == "crops/f_01_monitor.jpg" and rec["bbox"] == [10, 20, 210, 170]

    raw = (sessao / "inventario.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM para o Excel
    with (sessao / "inventario.csv").open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["rotulo_detector"], r["nome"], r["confere"]) for r in rows] == [
        ("monitor", "Monitor", "True"),
        ("cadeira", "Banqueta", "False"),
    ]
    assert rows[1]["categoria"] == "mobiliário"

    assert eventos == [{"type": "descriptions", "index": 3, "itens": ["Monitor Dell", "Banqueta (?)"]}]
    assert len(kf.extras["descriptions"]) == 2


def test_erro_do_modelo_nao_interrompe(tmp_path: Path):
    sessao, img = _sessao(tmp_path)
    proc = VlmProcessor(VlmSettings(_env_file=None), VlmFalso(falhar_em="monitor"))
    proc.setup()
    kf = Keyframe("s", 1, datetime.now(UTC), img, sessao / "f.jpg", 1.0, None, None,
                  extras={"detections": DETS})
    proc.process(kf)
    recs = [json.loads(x) for x in (sessao / "descriptions.jsonl").read_text().splitlines()]
    assert recs[0]["item"]["erro"] == "RuntimeError: CUDA out of memory"
    assert recs[1]["item"]["nome"] == "Banqueta"


def test_sem_yolo_antes_nao_faz_nada(tmp_path: Path):
    sessao, img = _sessao(tmp_path)
    vlm = VlmFalso()
    proc = VlmProcessor(VlmSettings(_env_file=None), vlm)
    proc.setup()
    proc.process(Keyframe("s", 1, datetime.now(UTC), img, sessao / "f.jpg", 1.0, None, None))
    assert vlm.chamadas == [] and not (sessao / "descriptions.jsonl").exists()


def test_prompt_personalizado(tmp_path: Path):
    arq = tmp_path / "prompt.txt"
    arq.write_text("Descreva o {rotulo} em JSON.", encoding="utf-8")
    vlm = VlmFalso()
    core = ItemDescriber(VlmSettings(_env_file=None, prompt_file=arq), vlm)
    core.load()
    sessao, img = _sessao(tmp_path)
    from inventario_visual.vision.vlm import DescriptionWriter

    list(core.describe_frame(img, DETS[:1], {"file": "f.jpg", "index": 1}, DescriptionWriter(sessao)))
    assert vlm.chamadas[0][1] == "Descreva o monitor em JSON."


def test_lote_descreve_retoma_e_refaz(tmp_path: Path):
    sessao, _ = _sessao(tmp_path)
    linha = {
        "index": 1, "file": "f.jpg", "captured_at": "2026-10-17T12:00:00+00:00", "counts": {},
        "detections": [
            {"label": d.label, "prompt": d.prompt, "confidence": d.confidence,
             "bbox": list(d.bbox), "crop_file": d.crop_file} for d in DETS
        ],
    }
    vazio = {**linha, "index": 2, "file": "g.jpg", "detections": []}
    (sessao / "detections.jsonl").write_text(json.dumps(linha) + "\n" + json.dumps(vazio) + "\n")

    vlm = VlmFalso()
    core = ItemDescriber(VlmSettings(_env_file=None), vlm)
    core.load()
    assert describe_session(sessao, core) == 2
    assert describe_session(sessao, core) == 0  # retoma: nada pendente
    assert len(vlm.chamadas) == 2
    assert describe_session(sessao, core, refazer=True) == 2
    assert len((sessao / "descriptions.jsonl").read_text().splitlines()) == 2
    with (sessao / "inventario.csv").open(encoding="utf-8-sig") as fh:
        assert len(list(csv.DictReader(fh))) == 2


def test_lote_sem_deteccoes(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        describe_session(tmp_path, ItemDescriber(VlmSettings(_env_file=None), VlmFalso()))
