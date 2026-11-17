"""Salvamento das fotos de cadastro.

Estrutura gerada:
    dados/integrantes/<nome>/<sessao>/alinhadas/0001.png   rosto 112x112 (entrada da CNN)
    dados/integrantes/<nome>/<sessao>/quadros/0001.jpg     quadro original inteiro

O quadro original é guardado para que dê para refazer o alinhamento depois
(por exemplo, com outro tamanho de recorte) sem precisar gravar de novo.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from src.config import DADOS_DIR

INTEGRANTES_DIR = DADOS_DIR / "integrantes"

# Rostos menores que isto (em pixels de largura) ficam ruins depois de alinhados.
LARGURA_MINIMA_ROSTO = 80


def normalizar_nome(texto: str) -> str:
    """'João da Silva' -> 'joao_da_silva' (seguro para usar como nome de pasta)."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sem_acento.lower()).strip("_")


def sessao_de_hoje() -> str:
    return date.today().isoformat()


class SessaoCaptura:
    """Pasta de uma pessoa numa sessão; numera as fotos sem sobrescrever."""

    def __init__(self, nome: str, sessao: str, raiz: Path = INTEGRANTES_DIR) -> None:
        self.nome = normalizar_nome(nome)
        self.sessao = normalizar_nome(sessao) or sessao_de_hoje()
        if not self.nome:
            raise ValueError("Informe o nome da pessoa.")
        self.pasta = raiz / self.nome / self.sessao
        self._alinhadas = self.pasta / "alinhadas"
        self._quadros = self.pasta / "quadros"

    def quantidade(self) -> int:
        if not self._alinhadas.exists():
            return 0
        return len(list(self._alinhadas.glob("*.png")))

    def _proximo_numero(self) -> int:
        existentes = [int(p.stem) for p in self._alinhadas.glob("*.png") if p.stem.isdigit()]
        return max(existentes, default=0) + 1

    def salvar(self, quadro_bgr: np.ndarray, rosto_alinhado: np.ndarray) -> Path:
        self._alinhadas.mkdir(parents=True, exist_ok=True)
        self._quadros.mkdir(parents=True, exist_ok=True)
        nome_arquivo = f"{self._proximo_numero():04d}"
        caminho = self._alinhadas / f"{nome_arquivo}.png"
        cv2.imwrite(str(caminho), rosto_alinhado)
        cv2.imwrite(str(self._quadros / f"{nome_arquivo}.jpg"), quadro_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        return caminho
