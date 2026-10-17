"""Configuração via variáveis de ambiente (prefixo ``INVENTARIO_``) ou arquivo ``.env``."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .keyframes import KeyframeSelectorConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INVENTARIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Servidor
    host: str = "0.0.0.0"
    port: int = 8443
    cert_file: Path = Path("certs/cert.pem")
    key_file: Path = Path("certs/key.pem")
    # Certificado da CA para instalar no celular (servido em /ca.crt).
    ca_file: Path = Path("certs/rootCA.crt")
    output_dir: Path = Path("data/keyframes")
    log_level: str = "info"

    # Seletor de frames-chave (ver KeyframeSelectorConfig)
    sample_fps: float = Field(default=5.0, gt=0)
    analysis_width: int = Field(default=640, ge=16)
    motion_threshold: float = Field(default=6.0, ge=0)
    stable_samples: int = Field(default=3, ge=1)
    sharpness_threshold: float = Field(default=80.0, ge=0)
    novelty_threshold: float = Field(default=12.0, ge=0)
    min_keyframe_interval_s: float = Field(default=1.0, ge=0)
    jpeg_quality: int = Field(default=92, ge=1, le=100)

    # Processadores plugáveis: "modulo:Classe,outro.modulo:Classe"
    processors: str = "inventario_visual.processors:LogProcessor"
    processor_queue_size: int = Field(default=32, ge=1)

    def selector_config(self) -> KeyframeSelectorConfig:
        return KeyframeSelectorConfig(
            sample_fps=self.sample_fps,
            analysis_width=self.analysis_width,
            motion_threshold=self.motion_threshold,
            stable_samples=self.stable_samples,
            sharpness_threshold=self.sharpness_threshold,
            novelty_threshold=self.novelty_threshold,
            min_keyframe_interval_s=self.min_keyframe_interval_s,
        )

    def processor_specs(self) -> list[str]:
        return [s.strip() for s in self.processors.split(",") if s.strip()]
