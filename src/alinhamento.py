"""Alinhamento facial: endireita e recorta o rosto em 112x112.

Usa os 5 pontos do detector (olhos, nariz e cantos da boca) para estimar uma
transformação de similaridade (rotação + escala + translação, sem distorcer)
que leva o rosto para as posições do modelo padrão do ArcFace. As CNNs de
reconhecimento pré-treinadas esperam exatamente esse recorte.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.deteccao import Deteccao

TAMANHO = 112

# Posições de referência (x, y) no recorte 112x112, na ordem do YuNet:
# olho à esquerda da imagem, olho à direita, nariz, boca à esquerda, boca à direita.
MODELO_ARCFACE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def matriz_alinhamento(pontos: np.ndarray, tamanho: int = TAMANHO) -> np.ndarray:
    """Matriz 2x3 de similaridade que leva os 5 pontos ao modelo de referência."""
    destino = MODELO_ARCFACE * (tamanho / TAMANHO)
    matriz, _ = cv2.estimateAffinePartial2D(
        pontos.astype(np.float32), destino, method=cv2.LMEDS
    )
    if matriz is None:
        raise ValueError("Não foi possível estimar o alinhamento a partir dos pontos.")
    return matriz


def alinhar(imagem_bgr: np.ndarray, deteccao: Deteccao, tamanho: int = TAMANHO) -> np.ndarray:
    """Devolve o rosto alinhado e recortado (tamanho x tamanho, BGR).

    Use a imagem original da câmera, não a espelhada: espelhar troca os lados
    dos olhos e da boca em relação ao modelo de referência.
    """
    matriz = matriz_alinhamento(deteccao.pontos, tamanho)
    return cv2.warpAffine(
        imagem_bgr, matriz, (tamanho, tamanho), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def angulo_olhos(deteccao: Deteccao) -> float:
    """Inclinação da linha dos olhos em graus (0 = cabeça reta)."""
    (x1, y1), (x2, y2) = deteccao.pontos[0], deteccao.pontos[1]
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
