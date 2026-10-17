"""Testes leves da aplicação FastAPI (sem WebRTC real)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from inventario_visual.app import create_app
from inventario_visual.config import Settings


def test_pagina_e_status(tmp_path: Path):
    settings = Settings(output_dir=tmp_path, ca_file=tmp_path / "nao-existe.crt", _env_file=None)
    with TestClient(create_app(settings)) as client:
        r = client.get("/")
        assert r.status_code == 200 and "getUserMedia" not in r.text and "app.js" in r.text
        js = client.get("/static/app.js")
        assert js.status_code == 200 and 'facingMode: { ideal: "environment" }' in js.text
        st = client.get("/api/status").json()
        assert st["sessions"] == {} and st["processors"] == ["log"]
        assert client.get("/ca.crt").status_code == 404
        assert client.post("/api/offer", json={"sdp": "x", "type": "answer"}).status_code == 400
