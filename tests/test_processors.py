"""Interface FrameProcessor, pipeline assíncrono e gravação em disco."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from inventario_visual.processors import (
    FrameProcessor,
    Keyframe,
    LogProcessor,
    ProcessorPipeline,
    load_processors,
)
from inventario_visual.storage import KeyframeStore

from .conftest import cena_texturizada


class Coletor(FrameProcessor):
    name = "coletor"

    def __init__(self) -> None:
        self.recebidos: list[int] = []
        self.setup_chamado = False
        self.close_chamado = False

    def setup(self) -> None:
        self.setup_chamado = True

    def process(self, keyframe: Keyframe) -> None:
        self.recebidos.append(keyframe.index)
        keyframe.extras["coletor"] = True


class Quebrado(FrameProcessor):
    name = "quebrado"

    def process(self, keyframe: Keyframe) -> None:
        raise RuntimeError("falha proposital")


def _kf(i: int) -> Keyframe:
    return Keyframe("s", i, datetime.now(UTC), cena_texturizada(i), None, 100.0, 0.5, None)


def test_frame_processor_eh_abstrato():
    with pytest.raises(TypeError):
        FrameProcessor()  # type: ignore[abstract]


def test_pipeline_entrega_em_ordem_e_isola_falhas():
    async def cenario() -> Coletor:
        col = Coletor()
        pipe = ProcessorPipeline([Quebrado(), col])
        await pipe.start()
        for i in range(1, 6):
            pipe.submit(_kf(i))
        await pipe.join()
        await pipe.stop()
        assert pipe.processed == 5
        return col

    col = asyncio.run(cenario())
    assert col.setup_chamado
    assert col.recebidos == [1, 2, 3, 4, 5]


def test_load_processors():
    procs = load_processors(
        [
            "inventario_visual.processors:LogProcessor",
            " ",
            "inventario_visual.processors.VisionProcessorStub",  # formato com ponto
        ]
    )
    assert isinstance(procs[0], LogProcessor)
    assert len(procs) == 2
    with pytest.raises(TypeError):
        load_processors(["inventario_visual.storage:KeyframeStore"])


def test_store_salva_com_timestamp(tmp_path: Path):
    store = KeyframeStore(tmp_path / "sessao")
    quando = datetime(2026, 10, 17, 14, 30, 5, 123456, tzinfo=UTC)
    salvo = store.save(cena_texturizada(1), quando, 7, {"sharpness": 123.4})
    assert salvo.path.name == "20261017_143005_123456_0007.jpg"
    assert salvo.path.is_file()
    linha = json.loads((tmp_path / "sessao" / "keyframes.jsonl").read_text(encoding="utf-8"))
    assert linha["index"] == 7 and linha["sharpness"] == 123.4
    assert re.match(r"2026-10-17T14:30:05", linha["captured_at"])
