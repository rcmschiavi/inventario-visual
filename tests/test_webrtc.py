"""Integração: um cliente aiortc envia vídeo sintético ao SessionManager real."""

from __future__ import annotations

import asyncio
import json
from fractions import Fraction
from pathlib import Path

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from inventario_visual.keyframes import KeyframeSelectorConfig
from inventario_visual.processors import FrameProcessor, Keyframe, ProcessorPipeline
from inventario_visual.webrtc import SessionManager

from .conftest import cena_texturizada


class CenaParada(VideoStreamTrack):
    """Trilha que envia sempre a mesma cena texturizada (câmera parada)."""

    def __init__(self) -> None:
        super().__init__()
        self.img = cena_texturizada(5, h=480, w=640)

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(self.img, format="bgr24")
        frame.pts, frame.time_base = pts, time_base or Fraction(1, 90000)
        return frame


class Sinal(FrameProcessor):
    name = "sinal"

    def __init__(self) -> None:
        self.recebidos: list[Keyframe] = []

    def process(self, keyframe: Keyframe) -> None:
        self.recebidos.append(keyframe)
        # Roda na thread de trabalho: testa o envio de eventos ao celular.
        keyframe.send_event({"type": "detections", "index": keyframe.index, "counts": {}})


def test_streaming_ponta_a_ponta(tmp_path: Path):
    async def cenario() -> tuple[Sinal, list[dict[str, object]]]:
        sinal = Sinal()
        pipeline = ProcessorPipeline([sinal])
        await pipeline.start()
        # Limiares folgados: a compressão VP8 suaviza a imagem.
        cfg = KeyframeSelectorConfig(sample_fps=10, sharpness_threshold=20, motion_threshold=8)
        manager = SessionManager(cfg, tmp_path, pipeline)

        cliente = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        eventos: list[dict[str, object]] = []
        canal = cliente.createDataChannel("eventos")
        canal.on("message", lambda m: eventos.append(json.loads(m)))
        cliente.addTrack(CenaParada())
        await cliente.setLocalDescription(await cliente.createOffer())

        sid, answer = await manager.handle_offer(cliente.localDescription.sdp, "offer")
        await cliente.setRemoteDescription(RTCSessionDescription(answer.sdp, answer.type))

        for _ in range(150):  # até 15 s
            await asyncio.sleep(0.1)
            if sinal.recebidos:
                break
        await asyncio.sleep(0.3)
        await cliente.close()
        await manager.close_all()
        await pipeline.stop()
        return sinal, eventos

    sinal, eventos = asyncio.run(cenario())
    assert sinal.recebidos, "nenhum frame-chave recebido via WebRTC"
    kf = sinal.recebidos[0]
    assert kf.image.shape == (480, 640, 3)
    assert kf.path is not None and kf.path.is_file()
    assert any(e.get("type") == "keyframe" for e in eventos)
    assert any(e.get("type") == "detections" for e in eventos)
