"""Testes do seletor de frames-chave com imagens/vídeos sintéticos."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from inventario_visual.keyframes import (
    Decision,
    KeyframeSelector,
    KeyframeSelectorConfig,
    laplacian_variance,
    mean_abs_diff,
    to_analysis_gray,
)

from .conftest import borrar, cena_texturizada, deslocar, ruido_sensor

FPS = 30.0
# Config de teste: amostra todo frame a 10 fps e exige 3 amostras estáveis.
CFG = KeyframeSelectorConfig(
    sample_fps=10.0,
    analysis_width=320,
    motion_threshold=6.0,
    stable_samples=3,
    sharpness_threshold=80.0,
    novelty_threshold=12.0,
    min_keyframe_interval_s=0.5,
)


def rodar(selector: KeyframeSelector, frames: list[np.ndarray], t0: float = 0.0, fps: float = 10.0):
    """Alimenta o seletor e devolve (resultados, instantes dos frames-chave)."""
    resultados = []
    for i, f in enumerate(frames):
        resultados.append(selector.update(f, t0 + i / fps))
    tempos = [t0 + i / fps for i, r in enumerate(resultados) if r.is_keyframe]
    return resultados, tempos


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #


def test_nitidez_maior_em_imagem_nitida(cena):
    g_nitida = to_analysis_gray(cena, 320)
    g_borrada = to_analysis_gray(borrar(cena), 320)
    assert laplacian_variance(g_nitida) > 10 * laplacian_variance(g_borrada)
    assert laplacian_variance(g_borrada) < CFG.sharpness_threshold < laplacian_variance(g_nitida)


def test_movimento_zero_para_imagens_iguais(cena):
    g = to_analysis_gray(cena, 320)
    assert mean_abs_diff(g, g) == 0.0
    assert mean_abs_diff(g, to_analysis_gray(deslocar(cena, 20), 320)) > CFG.motion_threshold


def test_reducao_de_resolucao_mantem_proporcao():
    img = np.zeros((1080, 1920, 3), np.uint8)
    assert to_analysis_gray(img, 640).shape == (360, 640)
    # Nunca amplia imagens menores que a largura de análise.
    assert to_analysis_gray(np.zeros((100, 200, 3), np.uint8), 640).shape == (100, 200)


def test_config_invalida():
    with pytest.raises(ValueError):
        KeyframeSelectorConfig(sample_fps=0)
    with pytest.raises(ValueError):
        KeyframeSelectorConfig(blur_kernel=4)


# --------------------------------------------------------------------------- #
# Seletor
# --------------------------------------------------------------------------- #


def test_cena_parada_e_nitida_gera_um_frame_chave(cena):
    sel = KeyframeSelector(CFG)
    frames = [ruido_sensor(cena, s) for s in range(15)]
    resultados, tempos = rodar(sel, frames)
    # Emite assim que completa 3 amostras estáveis, e só uma vez (sem duplicatas).
    assert len(tempos) == 1
    assert tempos[0] == pytest.approx(0.2)
    assert resultados[0].decision is Decision.STABILIZING
    depois = {r.decision for r in resultados[3:]}
    assert depois <= {Decision.COOLDOWN, Decision.DUPLICATE, Decision.STABILIZING}
    assert Decision.DUPLICATE in {r.decision for r in resultados}


def test_frame_chave_eh_imagem_original_em_resolucao_cheia():
    cena_hd = cena_texturizada(3, h=720, w=1280)
    sel = KeyframeSelector(CFG)
    resultados, _ = rodar(sel, [cena_hd] * 3)
    kf = resultados[-1]
    assert kf.is_keyframe
    assert kf.keyframe_image is not None and kf.keyframe_image.shape == (720, 1280, 3)
    assert kf.keyframe_metrics is not None and kf.keyframe_metrics.sharpness >= CFG.sharpness_threshold


def test_camera_em_movimento_nao_gera_frame_chave(cena):
    sel = KeyframeSelector(CFG)
    frames = [deslocar(cena, 12 * i) for i in range(20)]
    resultados, tempos = rodar(sel, frames)
    assert tempos == []
    assert all(r.decision is Decision.MOVING for r in resultados[1:])


def test_cena_borrada_nao_gera_frame_chave(cena):
    sel = KeyframeSelector(CFG)
    borrada = borrar(cena)
    resultados, tempos = rodar(sel, [borrada] * 10)
    assert tempos == []
    assert resultados[-1].decision is Decision.BLURRY


def test_escolhe_o_frame_mais_nitido_da_janela_estavel(cena):
    # Janela com foco variando levemente (pouco movimento entre eles).
    leve = borrar(cena, 3)
    sel = KeyframeSelector(replace(CFG, motion_threshold=30.0))
    resultados, _ = rodar(sel, [leve, cena, leve])
    kf = resultados[-1]
    assert kf.is_keyframe
    nit_cena = laplacian_variance(to_analysis_gray(cena, 320))
    assert kf.keyframe_metrics is not None
    assert kf.keyframe_metrics.sharpness == pytest.approx(nit_cena)
    assert np.array_equal(kf.keyframe_image, cena)


def test_video_sintetico_varias_cenas():
    """Simula: parar no item A, andar, parar no item B, andar, voltar a parar em B."""
    a, b = cena_texturizada(10), cena_texturizada(20)
    frames: list[np.ndarray] = []
    frames += [ruido_sensor(a, i) for i in range(8)]  # parado em A
    frames += [deslocar(a, 15 * i) for i in range(1, 6)]  # andando
    frames += [ruido_sensor(b, 100 + i) for i in range(8)]  # parado em B
    frames += [deslocar(b, 15 * i) for i in range(1, 4)]  # tremida
    frames += [ruido_sensor(b, 200 + i) for i in range(8)]  # B de novo -> duplicata
    sel = KeyframeSelector(CFG)
    _, tempos = rodar(sel, frames)
    assert len(tempos) == 2
    assert sel.keyframes_emitted == 2


def test_intervalo_minimo_entre_frames_chave():
    a, b = cena_texturizada(30), cena_texturizada(31)
    cfg = replace(CFG, stable_samples=1, min_keyframe_interval_s=1.0, motion_threshold=255.0)
    sel = KeyframeSelector(cfg)
    # Alterna cenas diferentes a cada 0,1 s: todas seriam novas, mas o cooldown limita.
    frames = [a if i % 2 == 0 else b for i in range(25)]
    resultados, tempos = rodar(sel, frames)
    assert len(tempos) >= 2
    assert all(t2 - t1 >= 1.0 - 1e-9 for t1, t2 in zip(tempos, tempos[1:], strict=False))
    assert Decision.COOLDOWN in {r.decision for r in resultados}


def test_amostragem_descarta_frames_extras(cena):
    """Alimentado a 30 fps com sample_fps=10, só ~1/3 dos frames é analisado."""
    sel = KeyframeSelector(CFG)
    resultados, _ = rodar(sel, [cena] * 30, fps=FPS)
    analisados = [r for r in resultados if r.decision is not Decision.SKIPPED]
    assert 9 <= len(analisados) <= 11


def test_reset_esquece_estado(cena):
    sel = KeyframeSelector(CFG)
    rodar(sel, [cena] * 3)
    assert sel.keyframes_emitted == 1
    sel.reset()
    _, tempos = rodar(sel, [cena] * 3, t0=100.0)
    assert len(tempos) == 1
