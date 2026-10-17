"""Aplicação FastAPI: serve a página do celular e recebe a oferta WebRTC."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings
from .processors import ProcessorPipeline, load_processors
from .webrtc import SessionManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(str(resources.files("inventario_visual") / "static"))


class Offer(BaseModel):
    sdp: str
    type: str


class Answer(BaseModel):
    sdp: str
    type: str
    session_id: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pipeline = ProcessorPipeline(
            load_processors(settings.processor_specs()), max_queue=settings.processor_queue_size
        )
        await pipeline.start()
        manager = SessionManager(
            settings.selector_config(), settings.output_dir, pipeline, settings.jpeg_quality
        )
        app.state.pipeline = pipeline
        app.state.manager = manager
        try:
            yield
        finally:
            await manager.close_all()
            await pipeline.stop()

    app = FastAPI(title="inventario-visual", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.post("/api/offer", response_model=Answer)
    async def offer(body: Offer) -> Answer:
        if body.type != "offer":
            raise HTTPException(400, "type deve ser 'offer'")
        manager: SessionManager = app.state.manager
        session_id, local = await manager.handle_offer(body.sdp, body.type)
        return Answer(sdp=local.sdp, type=local.type, session_id=session_id)

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def close_session(session_id: str) -> None:
        manager: SessionManager = app.state.manager
        await manager.close_session(session_id)

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        manager: SessionManager = app.state.manager
        pipeline: ProcessorPipeline = app.state.pipeline
        return {
            "sessions": {
                sid: {
                    "state": s.pc.connectionState,
                    "frames_received": s.stats.frames_received,
                    "frames_analyzed": s.stats.frames_analyzed,
                    "keyframes": s.stats.keyframes,
                    "last_decision": s.stats.last_decision,
                    "resolution": f"{s.stats.width}x{s.stats.height}",
                }
                for sid, s in manager.sessions.items()
            },
            "processors": [p.name for p in pipeline.processors],
            "processed": pipeline.processed,
            "dropped": pipeline.dropped,
            "selector": settings.selector_config().__repr__(),
        }

    @app.get("/ca.crt")
    async def ca_certificate() -> FileResponse:
        """Baixa a CA local para instalar no celular (ver README)."""
        if not settings.ca_file.is_file():
            raise HTTPException(404, "CA não encontrada; rode scripts/gerar_certificado.py")
        return FileResponse(
            settings.ca_file,
            media_type="application/x-x509-ca-cert",
            filename="inventario-visual-ca.crt",
        )

    return app
