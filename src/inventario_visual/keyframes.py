"""Seletor de frames-chave.

A ideia é simples: o vídeo chega a ~30 fps, mas só interessa guardar os
momentos em que o usuário "parou" a câmera sobre algo e a imagem está nítida.

Pipeline por frame amostrado:

1. Reduz a resolução e converte para tons de cinza (análise barata).
2. Mede o **movimento**: diferença absoluta média em relação ao frame
   amostrado anterior.
3. Mede a **nitidez**: variância do Laplaciano.
4. Se houver ``stable_samples`` amostras seguidas com pouco movimento, emite o
   frame mais nítido dessa janela estável — desde que ele passe no limiar de
   nitidez, respeite o intervalo mínimo entre frames-chave e seja
   suficientemente diferente do último frame-chave (novidade).

O seletor é puramente síncrono e não conhece WebRTC, o que permite testá-lo
com imagens sintéticas e reaproveitá-lo para arquivos de vídeo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

# Imagem BGR uint8 (formato padrão do OpenCV).
ImageBGR = NDArray[np.uint8]
ImageGray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class KeyframeSelectorConfig:
    """Parâmetros do seletor (todos configuráveis via variáveis de ambiente)."""

    # Quantos frames por segundo analisar (o resto é descartado).
    sample_fps: float = 5.0
    # Largura usada na análise; a altura segue a proporção original.
    analysis_width: int = 640
    # Diferença absoluta média (0-255) abaixo da qual a cena é considerada parada.
    motion_threshold: float = 6.0
    # Quantas amostras estáveis consecutivas são exigidas antes de emitir.
    stable_samples: int = 3
    # Variância mínima do Laplaciano (na resolução de análise) para ser "nítido".
    sharpness_threshold: float = 80.0
    # Diferença mínima (0-255) em relação ao último frame-chave para evitar duplicatas.
    novelty_threshold: float = 12.0
    # Intervalo mínimo, em segundos, entre dois frames-chave.
    min_keyframe_interval_s: float = 1.0
    # Desfoque aplicado antes da medida de movimento (reduz ruído do sensor).
    blur_kernel: int = 5

    def __post_init__(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("sample_fps deve ser > 0")
        if self.analysis_width < 16:
            raise ValueError("analysis_width deve ser >= 16")
        if self.stable_samples < 1:
            raise ValueError("stable_samples deve ser >= 1")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel deve ser ímpar e >= 1")

    @property
    def sample_interval_s(self) -> float:
        return 1.0 / self.sample_fps


class Decision(StrEnum):
    """Resultado da análise de um frame (útil para depuração e calibração)."""

    SKIPPED = "skipped"  # frame não amostrado (taxa de amostragem)
    MOVING = "moving"  # muito movimento
    STABILIZING = "stabilizing"  # parado, mas ainda acumulando amostras
    BLURRY = "blurry"  # parado, porém sem nitidez suficiente
    COOLDOWN = "cooldown"  # intervalo mínimo ainda não passou
    DUPLICATE = "duplicate"  # muito parecido com o último frame-chave
    KEYFRAME = "keyframe"  # frame-chave emitido


@dataclass(slots=True)
class FrameMetrics:
    """Métricas calculadas para um frame amostrado."""

    timestamp: float
    motion: float | None  # None no primeiro frame (sem referência)
    sharpness: float


@dataclass(slots=True)
class SelectionResult:
    """Retorno de :meth:`KeyframeSelector.update`."""

    decision: Decision
    metrics: FrameMetrics | None = None
    # Preenchidos somente quando ``decision == Decision.KEYFRAME``.
    keyframe_image: ImageBGR | None = None
    keyframe_metrics: FrameMetrics | None = None
    novelty: float | None = None

    @property
    def is_keyframe(self) -> bool:
        return self.decision is Decision.KEYFRAME


# --------------------------------------------------------------------------- #
# Métricas (funções puras, testáveis isoladamente)
# --------------------------------------------------------------------------- #


def to_analysis_gray(image: ImageBGR, width: int) -> ImageGray:
    """Converte para cinza e reduz para ``width`` pixels de largura (sem ampliar)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]
    if w > width:
        new_h = max(1, round(h * width / w))
        gray = np.asarray(cv2.resize(gray, (width, new_h), interpolation=cv2.INTER_AREA))
    return np.ascontiguousarray(gray, dtype=np.uint8)


def laplacian_variance(gray: ImageGray) -> float:
    """Nitidez: variância do Laplaciano. Imagens borradas têm valores baixos."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def mean_abs_diff(a: ImageGray, b: ImageGray) -> float:
    """Diferença absoluta média entre duas imagens em cinza (escala 0-255)."""
    if a.shape != b.shape:
        b = np.asarray(cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA))
    return float(cv2.absdiff(a, b).mean())


# --------------------------------------------------------------------------- #
# Seletor
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Candidate:
    image: ImageBGR
    gray: ImageGray
    metrics: FrameMetrics


@dataclass(slots=True)
class KeyframeSelector:
    """Máquina de estados que decide quais frames são frames-chave.

    Uso::

        selector = KeyframeSelector(KeyframeSelectorConfig())
        for ts, frame in frames:
            result = selector.update(frame, ts)
            if result.is_keyframe:
                salvar(result.keyframe_image)

    ``timestamp`` é em segundos e deve ser monotônico (ex.: ``time.monotonic()``
    ou o PTS do vídeo). Passá-lo explicitamente deixa os testes determinísticos.
    """

    config: KeyframeSelectorConfig = field(default_factory=KeyframeSelectorConfig)

    _last_sample_ts: float | None = None
    _prev_blurred: ImageGray | None = None
    _stable_count: int = 0
    _best: _Candidate | None = None
    _last_keyframe_gray: ImageGray | None = None
    _last_keyframe_ts: float | None = None
    keyframes_emitted: int = 0

    def reset(self) -> None:
        """Esquece todo o estado (ex.: nova sessão de câmera)."""
        self._last_sample_ts = None
        self._prev_blurred = None
        self._stable_count = 0
        self._best = None
        self._last_keyframe_gray = None
        self._last_keyframe_ts = None
        self.keyframes_emitted = 0

    def should_sample(self, timestamp: float) -> bool:
        """Indica se o frame neste instante será analisado.

        Permite ao chamador evitar converter o frame (custo de decodificação para
        ndarray) quando ele seria descartado de qualquer forma.
        """
        if self._last_sample_ts is None:
            return True
        # Pequena tolerância para não perder amostras por jitter de timestamp.
        return timestamp - self._last_sample_ts >= self.config.sample_interval_s * 0.95

    def update(self, image: ImageBGR, timestamp: float) -> SelectionResult:
        """Processa um frame e informa se ele gerou um frame-chave."""
        if not self.should_sample(timestamp):
            return SelectionResult(Decision.SKIPPED)
        self._last_sample_ts = timestamp

        cfg = self.config
        gray = to_analysis_gray(image, cfg.analysis_width)
        blurred: ImageGray = (
            np.asarray(cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0), dtype=np.uint8)
            if cfg.blur_kernel > 1
            else gray
        )
        motion = None if self._prev_blurred is None else mean_abs_diff(blurred, self._prev_blurred)
        self._prev_blurred = blurred
        metrics = FrameMetrics(timestamp=timestamp, motion=motion, sharpness=laplacian_variance(gray))

        # Primeiro frame ou câmera em movimento: zera a janela estável.
        if motion is None or motion > cfg.motion_threshold:
            self._stable_count = 0
            self._best = None
            # O primeiro frame conta como início de uma possível janela estável.
            if motion is None:
                self._stable_count = 1
                self._best = _Candidate(image, gray, metrics)
                return self._try_emit(metrics)
            return SelectionResult(Decision.MOVING, metrics)

        # Cena parada: acumula e guarda o frame mais nítido da janela.
        self._stable_count += 1
        if self._best is None or metrics.sharpness > self._best.metrics.sharpness:
            self._best = _Candidate(image, gray, metrics)
        return self._try_emit(metrics)

    def _try_emit(self, metrics: FrameMetrics) -> SelectionResult:
        cfg = self.config
        best = self._best
        if best is None or self._stable_count < cfg.stable_samples:
            return SelectionResult(Decision.STABILIZING, metrics)

        if best.metrics.sharpness < cfg.sharpness_threshold:
            return SelectionResult(Decision.BLURRY, metrics)

        if (
            self._last_keyframe_ts is not None
            and metrics.timestamp - self._last_keyframe_ts < cfg.min_keyframe_interval_s
        ):
            return SelectionResult(Decision.COOLDOWN, metrics)

        novelty: float | None = None
        if self._last_keyframe_gray is not None:
            novelty = mean_abs_diff(best.gray, self._last_keyframe_gray)
            if novelty < cfg.novelty_threshold:
                return SelectionResult(Decision.DUPLICATE, metrics, novelty=novelty)

        # Emite! Reinicia a janela para exigir nova estabilização.
        self._last_keyframe_gray = best.gray
        self._last_keyframe_ts = metrics.timestamp
        self._stable_count = 0
        self._best = None
        self.keyframes_emitted += 1
        return SelectionResult(
            Decision.KEYFRAME,
            metrics,
            keyframe_image=best.image,
            keyframe_metrics=best.metrics,
            novelty=novelty,
        )
