"""Interface plugável para processar frames-chave.

O streaming (WebRTC) e o seletor nunca chamam modelos de visão diretamente.
Eles apenas entregam cada :class:`Keyframe` a uma lista de
:class:`FrameProcessor`. Para adicionar YOLO-World + modelo de
visão-linguagem, basta criar uma subclasse e registrá-la em
``INVENTARIO_PROCESSORS`` — sem tocar no código de streaming.

Os processadores rodam numa thread de trabalho dedicada (ver
:class:`ProcessorPipeline`), então podem ser lentos e bloqueantes (GPU,
inferência síncrona) sem travar o recebimento de vídeo.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .keyframes import ImageBGR

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Keyframe:
    """Frame-chave já salvo em disco, entregue aos processadores."""

    session_id: str
    index: int  # sequencial dentro da sessão (1, 2, 3...)
    captured_at: datetime  # horário de parede (UTC) da captura
    image: ImageBGR  # imagem em resolução original, BGR uint8
    path: Path | None  # arquivo JPEG salvo (None se o salvamento falhou)
    sharpness: float
    motion: float | None
    novelty: float | None
    # Espaço livre para processadores anteriores deixarem resultados
    # para os seguintes (ex.: detecções do YOLO usadas pelo VLM).
    extras: dict[str, Any] = field(default_factory=dict)
    # Envia um evento JSON ao celular que gerou o frame (seguro para chamar da
    # thread de trabalho). None quando não há celular conectado (ex.: offline).
    notify: Callable[[dict[str, Any]], None] | None = None

    def send_event(self, payload: dict[str, Any]) -> None:
        if self.notify is not None:
            self.notify(payload)


class FrameProcessor(ABC):
    """Classe base para qualquer etapa que consome frames-chave.

    Ciclo de vida:

    * :meth:`setup` — chamado uma vez, na thread de trabalho, antes do
      primeiro frame. Bom lugar para carregar pesos na GPU.
    * :meth:`process` — chamado para cada frame-chave, em ordem.
    * :meth:`close` — chamado ao desligar o servidor.

    Exceções em :meth:`process` são registradas no log e não derrubam o
    pipeline nem os demais processadores.
    """

    #: Nome amigável usado nos logs.
    name: str = "processor"

    def setup(self) -> None:  # noqa: B027 - implementação opcional
        """Inicialização pesada (opcional)."""

    @abstractmethod
    def process(self, keyframe: Keyframe) -> None:
        """Processa um frame-chave. Pode bloquear (roda fora do event loop)."""

    def close(self) -> None:  # noqa: B027 - implementação opcional
        """Liberação de recursos (opcional)."""


class LogProcessor(FrameProcessor):
    """Processador de exemplo: apenas registra o frame no log."""

    name = "log"

    def process(self, keyframe: Keyframe) -> None:
        h, w = keyframe.image.shape[:2]
        logger.info(
            "[%s] frame-chave #%d %dx%d nitidez=%.1f -> %s",
            keyframe.session_id,
            keyframe.index,
            w,
            h,
            keyframe.sharpness,
            keyframe.path,
        )


class VisionProcessorStub(FrameProcessor):
    """Esqueleto da futura etapa de visão (YOLO-World + VLM).

    Exemplo de como ficará (não implementado aqui de propósito)::

        def setup(self) -> None:
            from ultralytics import YOLOWorld
            self.detector = YOLOWorld("yolov8l-worldv2.pt").to("cuda")
            self.detector.set_classes(["caixa", "garrafa", "ferramenta"])

        def process(self, keyframe: Keyframe) -> None:
            dets = self.detector.predict(keyframe.image, verbose=False)
            keyframe.extras["detections"] = dets
            # ... recortar caixas e enviar ao modelo de visão-linguagem ...
    """

    name = "vision-stub"

    def process(self, keyframe: Keyframe) -> None:
        keyframe.extras.setdefault("vision", "não implementado")


def load_processors(specs: Iterable[str]) -> list[FrameProcessor]:
    """Instancia processadores a partir de ``"pacote.modulo:Classe"``.

    Também aceita ``"pacote.modulo.Classe"`` (último ponto separa a classe).
    """
    processors: list[FrameProcessor] = []
    for raw in specs:
        spec = raw.strip()
        if not spec:
            continue
        module_name, sep, class_name = spec.partition(":")
        if not sep:
            module_name, _, class_name = spec.rpartition(".")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not (isinstance(cls, type) and issubclass(cls, FrameProcessor)):
            raise TypeError(f"{spec} não é subclasse de FrameProcessor")
        processors.append(cls())
    return processors


_STOP = object()


class ProcessorPipeline:
    """Fila assíncrona que entrega frames-chave aos processadores em ordem.

    Uma única tarefa consumidora executa os processadores via
    :func:`asyncio.to_thread`, preservando a ordem dos frames. A fila é
    limitada: se a visão ficar para trás, os frames mais antigos ainda não
    processados são descartados (eles continuam salvos em disco).
    """

    def __init__(self, processors: Sequence[FrameProcessor], max_queue: int = 32) -> None:
        self.processors = list(processors)
        self._queue: asyncio.Queue[object] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task[None] | None = None
        self.processed = 0
        self.dropped = 0

    async def start(self) -> None:
        for proc in self.processors:
            await asyncio.to_thread(proc.setup)
            logger.info("Processador carregado: %s", proc.name)
        self._task = asyncio.create_task(self._run(), name="processor-pipeline")

    def submit(self, keyframe: Keyframe) -> None:
        """Enfileira sem bloquear; descarta o mais antigo se a fila estiver cheia."""
        if not self.processors:
            return
        while True:
            try:
                self._queue.put_nowait(keyframe)
                return
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self.dropped += 1
                    logger.warning("Fila de processamento cheia; frame antigo descartado")

    async def join(self) -> None:
        """Aguarda até a fila esvaziar (útil em testes)."""
        await self._queue.join()

    async def stop(self) -> None:
        if self._task is not None:
            await self._queue.put(_STOP)
            await self._task
            self._task = None
        for proc in self.processors:
            try:
                await asyncio.to_thread(proc.close)
            except Exception:
                logger.exception("Erro ao fechar processador %s", proc.name)

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, Keyframe)
                await asyncio.to_thread(self._process_all, item)
                self.processed += 1
            finally:
                self._queue.task_done()

    def _process_all(self, keyframe: Keyframe) -> None:
        for proc in self.processors:
            try:
                proc.process(keyframe)
            except Exception:
                logger.exception("Processador %s falhou no frame #%d", proc.name, keyframe.index)
