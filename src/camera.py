"""Leitura de webcam com atraso mínimo.

O OpenCV guarda uma fila de quadros. Se o processamento for mais lento que a
câmera, a fila enche e a tela passa a mostrar quadros velhos (atraso que só
cresce). Aqui uma thread lê a câmera sem parar e guarda apenas o quadro mais
recente, então quem consome sempre recebe a imagem atual.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np


def abrir_captura(indice: int) -> cv2.VideoCapture:
    """Abre a câmera com o backend nativo do sistema e buffer mínimo."""
    if sys.platform.startswith("linux"):
        captura = cv2.VideoCapture(indice, cv2.CAP_V4L2)  # evita o backend FFMPEG, que acumula quadros
    elif sys.platform == "win32":
        captura = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    else:
        captura = cv2.VideoCapture(indice)
    captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return captura


class CameraAoVivo:
    """Câmera lida em segundo plano; `ler()` devolve sempre o quadro mais novo."""

    def __init__(self, indice: int) -> None:
        self.indice = indice
        self._captura = abrir_captura(indice)
        if not self._captura.isOpened():
            raise RuntimeError(f"Não consegui abrir a câmera {indice}.")

        self._trava = threading.Lock()
        self._quadro: np.ndarray | None = None
        self._instante = 0.0      # time.perf_counter() de quando o quadro chegou
        self._numero = 0          # contador para saber se há quadro novo
        self._rodando = True
        self._thread = threading.Thread(target=self._ler_sem_parar, daemon=True)
        self._thread.start()

    def _ler_sem_parar(self) -> None:
        while self._rodando:
            ok, quadro = self._captura.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._trava:
                self._quadro = quadro
                self._instante = time.perf_counter()
                self._numero += 1

    def ler(self) -> tuple[int, np.ndarray | None, float]:
        """Devolve (número do quadro, quadro, instante de chegada).

        O número só muda quando chega um quadro novo; compare com o anterior
        para não processar a mesma imagem duas vezes.
        """
        with self._trava:
            return self._numero, self._quadro, self._instante

    def fechar(self) -> None:
        self._rodando = False
        self._thread.join(timeout=1.0)
        self._captura.release()

    def __enter__(self) -> "CameraAoVivo":
        return self

    def __exit__(self, *_exc) -> None:
        self.fechar()
