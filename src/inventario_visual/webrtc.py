"""Recebimento de vídeo via WebRTC (aiortc) e integração com o seletor.

Cada celular conectado vira uma :class:`CameraSession`. A sessão consome a
trilha de vídeo continuamente (para não acumular atraso), analisa apenas os
frames amostrados e, quando o seletor emite um frame-chave, salva em disco e
o encaminha ao :class:`~inventario_visual.processors.ProcessorPipeline`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from aiortc import RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import VideoFrame

from .keyframes import ImageBGR, KeyframeSelector, KeyframeSelectorConfig, SelectionResult
from .processors import Keyframe, ProcessorPipeline
from .storage import KeyframeStore

logger = logging.getLogger(__name__)


def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(3)


@dataclass(slots=True)
class SessionStats:
    frames_received: int = 0
    frames_analyzed: int = 0
    keyframes: int = 0
    last_decision: str = ""
    last_sharpness: float | None = None
    last_motion: float | None = None
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class CameraSession:
    session_id: str
    pc: RTCPeerConnection
    selector: KeyframeSelector
    store: KeyframeStore
    pipeline: ProcessorPipeline
    stats: SessionStats = field(default_factory=SessionStats)
    channel: RTCDataChannel | None = None
    task: asyncio.Task[None] | None = None

    def send_event(self, payload: dict[str, Any]) -> None:
        """Envia um evento JSON ao celular pelo DataChannel (se aberto)."""
        ch = self.channel
        if ch is not None and ch.readyState == "open":
            with contextlib.suppress(Exception):
                ch.send(json.dumps(payload))

    async def consume(self, track: MediaStreamTrack) -> None:
        """Laço principal: lê frames, amostra, seleciona e salva frames-chave."""
        last_status = 0.0
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                logger.info("[%s] trilha de vídeo encerrada", self.session_id)
                return
            if not isinstance(frame, VideoFrame):
                continue
            self.stats.frames_received += 1
            now = time.monotonic()

            # Descarta cedo os frames fora da taxa de amostragem (evita a
            # conversão YUV->BGR, que é o custo dominante).
            if not self.selector.should_sample(now):
                continue

            image: ImageBGR = np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)
            self.stats.width, self.stats.height = image.shape[1], image.shape[0]
            # Análise em thread para não travar o event loop (outras sessões, HTTP).
            result = await asyncio.to_thread(self.selector.update, image, now)
            self.stats.frames_analyzed += 1
            self.stats.last_decision = result.decision.value
            if result.metrics is not None:
                self.stats.last_sharpness = result.metrics.sharpness
                self.stats.last_motion = result.metrics.motion

            if result.is_keyframe and result.keyframe_image is not None:
                await self._handle_keyframe(result.keyframe_image, result)

            # Status periódico para o celular mostrar o que o seletor está vendo.
            if now - last_status >= 0.5:
                last_status = now
                self.send_event(
                    {
                        "type": "status",
                        "decision": self.stats.last_decision,
                        "sharpness": self.stats.last_sharpness,
                        "motion": self.stats.last_motion,
                        "keyframes": self.stats.keyframes,
                        "resolution": f"{self.stats.width}x{self.stats.height}",
                    }
                )

    async def _handle_keyframe(self, image: ImageBGR, result: SelectionResult) -> None:
        km = result.keyframe_metrics
        assert km is not None
        index = self.stats.keyframes + 1
        captured_at = datetime.now(UTC)
        path: Path | None = None
        try:
            saved = await asyncio.to_thread(
                self.store.save,
                image,
                captured_at,
                index,
                {
                    "sharpness": round(km.sharpness, 2),
                    "motion": None if km.motion is None else round(km.motion, 3),
                    "novelty": None if result.novelty is None else round(result.novelty, 3),
                },
            )
            path = saved.path
        except Exception:
            logger.exception("[%s] falha ao salvar frame-chave", self.session_id)
        self.stats.keyframes = index
        logger.info("[%s] frame-chave #%d salvo em %s", self.session_id, index, path)
        self.send_event({"type": "keyframe", "index": index, "file": path.name if path else None})
        self.pipeline.submit(
            Keyframe(
                session_id=self.session_id,
                index=index,
                captured_at=captured_at,
                image=image,
                path=path,
                sharpness=km.sharpness,
                motion=km.motion,
                novelty=result.novelty,
            )
        )

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task
        await self.pc.close()


class SessionManager:
    """Cria e encerra sessões WebRTC."""

    def __init__(
        self,
        selector_config: KeyframeSelectorConfig,
        output_dir: Path,
        pipeline: ProcessorPipeline,
        jpeg_quality: int = 92,
    ) -> None:
        self.selector_config = selector_config
        self.output_dir = output_dir
        self.pipeline = pipeline
        self.jpeg_quality = jpeg_quality
        self.sessions: dict[str, CameraSession] = {}

    async def handle_offer(self, sdp: str, type_: str) -> tuple[str, RTCSessionDescription]:
        """Recebe a oferta SDP do navegador e devolve a resposta do servidor."""
        # iceServers vazio: na rede local não precisamos de STUN externo
        # (o aiortc usaria o STUN do Google por padrão).
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        session_id = _new_session_id()
        session = CameraSession(
            session_id=session_id,
            pc=pc,
            selector=KeyframeSelector(self.selector_config),
            store=KeyframeStore(self.output_dir / session_id, self.jpeg_quality),
            pipeline=self.pipeline,
        )
        self.sessions[session_id] = session
        logger.info("[%s] nova conexão", session_id)

        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            session.channel = channel
            channel.send(json.dumps({"type": "hello", "session": session_id}))

        @pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind != "video":
                return
            logger.info("[%s] recebendo vídeo", session_id)
            session.task = asyncio.create_task(session.consume(track), name=f"consume-{session_id}")

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            logger.info("[%s] estado da conexão: %s", session_id, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await self.close_session(session_id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        # setLocalDescription do aiortc só retorna após coletar os candidatos ICE,
        # então a resposta já vai completa (sem trickle ICE).
        return session_id, pc.localDescription

    async def close_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is not None:
            await session.close()
            logger.info(
                "[%s] sessão encerrada (%d frames, %d frames-chave)",
                session_id,
                session.stats.frames_received,
                session.stats.keyframes,
            )

    async def close_all(self) -> None:
        await asyncio.gather(*(self.close_session(sid) for sid in list(self.sessions)))
