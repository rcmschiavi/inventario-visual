"""Roda o seletor sobre um arquivo de vídeo — útil para calibrar os limiares.

Exemplo::

    inventario-offline gravacao.mp4 --saida data/teste --verbose
    inventario-offline gravacao.mp4 --saida data/teste --processar   # + YOLO-World etc.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .keyframes import Decision, KeyframeSelector, KeyframeSelectorConfig
from .processors import FrameProcessor, Keyframe, load_processors
from .storage import KeyframeStore


def process_video(
    video_path: Path,
    config: KeyframeSelectorConfig,
    output_dir: Path | None = None,
    verbose: bool = False,
    processors: list[FrameProcessor] | None = None,
) -> list[float]:
    """Retorna os instantes (s) dos frames-chave; salva-os se ``output_dir`` for dado.

    Se ``processors`` for informado, cada frame-chave também passa por eles
    (de forma síncrona, na ordem), como aconteceria no servidor.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Não foi possível abrir {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    selector = KeyframeSelector(config)
    store = KeyframeStore(output_dir, 92) if output_dir else None
    start = datetime.now(UTC)
    times: list[float] = []
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = n / fps
            n += 1
            result = selector.update(np.asarray(frame, dtype=np.uint8), ts)
            if verbose and result.decision is not Decision.SKIPPED and result.metrics:
                m = result.metrics
                motion = "   -  " if m.motion is None else f"{m.motion:6.2f}"
                print(f"t={ts:7.2f}s mov={motion} nit={m.sharpness:8.1f} {result.decision.value}")
            if result.is_keyframe and result.keyframe_image is not None:
                times.append(ts)
                km = result.keyframe_metrics
                captured_at = start + timedelta(seconds=ts)
                path = None
                if store is not None:
                    path = store.save(
                        result.keyframe_image,
                        captured_at,
                        len(times),
                        {"video_time_s": round(ts, 3), "sharpness": km.sharpness if km else None},
                    ).path
                # O mesmo Keyframe passa por todos: cada processador enxerga o que
                # os anteriores deixaram em `extras` (ex.: detecções -> VLM).
                keyframe = Keyframe(
                    session_id=output_dir.name if output_dir else video_path.stem,
                    index=len(times),
                    captured_at=captured_at,
                    image=result.keyframe_image,
                    path=path,
                    sharpness=km.sharpness if km else 0.0,
                    motion=km.motion if km else None,
                    novelty=result.novelty,
                )
                for proc in processors or []:
                    proc.process(keyframe)
    finally:
        cap.release()
    return times


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seleciona frames-chave de um arquivo de vídeo")
    parser.add_argument("video", type=Path)
    parser.add_argument("--saida", type=Path, help="pasta onde salvar os frames-chave")
    parser.add_argument("--verbose", "-v", action="store_true", help="mostra métricas por amostra")
    parser.add_argument("--motion", type=float, help="sobrescreve motion_threshold")
    parser.add_argument("--sharpness", type=float, help="sobrescreve sharpness_threshold")
    parser.add_argument(
        "--processar",
        action="store_true",
        help="passa os frames-chave pelos processadores de INVENTARIO_PROCESSORS (ex.: YOLO-World)",
    )
    args = parser.parse_args(argv)
    if args.processar and args.saida is None:
        args.saida = Path("data/offline") / args.video.stem

    settings = Settings()
    cfg = settings.selector_config()
    if args.motion is not None:
        cfg = replace(cfg, motion_threshold=args.motion)
    if args.sharpness is not None:
        cfg = replace(cfg, sharpness_threshold=args.sharpness)

    processors = load_processors(settings.processor_specs()) if args.processar else []
    for proc in processors:
        proc.setup()
    try:
        times = process_video(args.video, cfg, args.saida, args.verbose, processors)
    finally:
        for proc in processors:
            proc.close()
    print(f"{len(times)} frame(s)-chave: " + ", ".join(f"{t:.2f}s" for t in times))
    if args.saida:
        print(f"Resultados em {args.saida}")


if __name__ == "__main__":
    main()
