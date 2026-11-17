"""Detecção facial com o YuNet (OpenCV).

Etapa pronta do pipeline, não é contribuição do trabalho. Para cada rosto
encontrado devolve a caixa, os 5 pontos de referência (usados depois no
alinhamento) e a confiança.

Uso pela linha de comando (a partir da raiz do projeto):
    python -m src.deteccao                  # primeira webcam com imagem
    python -m src.deteccao --camera 2       # outra webcam
    python -m src.deteccao --imagem foto.jpg
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.camera import CameraAoVivo, abrir_captura
from src.config import YUNET_ARQUIVO, YUNET_URL

# Ordem dos pontos de referência devolvidos pelo YuNet.
NOMES_PONTOS = ("olho_direito", "olho_esquerdo", "nariz", "boca_direita", "boca_esquerda")


@dataclass
class Deteccao:
    caixa: np.ndarray    # (4,) x, y, largura, altura em pixels
    pontos: np.ndarray   # (5, 2) coordenadas (x, y), na ordem de NOMES_PONTOS
    confianca: float

    @property
    def area(self) -> float:
        return float(self.caixa[2] * self.caixa[3])


def garantir_modelo(caminho: Path = YUNET_ARQUIVO, url: str = YUNET_URL) -> Path:
    """Baixa o modelo ONNX do YuNet se ele ainda não estiver em modelos/."""
    if not caminho.exists():
        caminho.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando o YuNet para {caminho} ...")
        temporario = caminho.with_suffix(".part")
        urllib.request.urlretrieve(url, temporario)
        temporario.rename(caminho)
    return caminho


class DetectorFacial:
    """Envoltório do cv2.FaceDetectorYN.

    confianca_minima: descarta detecções abaixo deste escore (0 a 1).
    limiar_nms: sobreposição máxima entre caixas antes de suprimir duplicatas.
    """

    def __init__(
        self,
        confianca_minima: float = 0.9,
        limiar_nms: float = 0.3,
        max_rostos: int = 5000,
        modelo: Path | None = None,
    ) -> None:
        caminho = modelo or garantir_modelo()
        self._detector = cv2.FaceDetectorYN.create(
            str(caminho), "", (320, 320), confianca_minima, limiar_nms, max_rostos
        )
        self._tamanho = (320, 320)

    def definir_confianca(self, confianca_minima: float) -> None:
        """Altera a confiança mínima sem recriar o detector."""
        self._detector.setScoreThreshold(float(confianca_minima))

    def detectar(self, imagem_bgr: np.ndarray) -> list[Deteccao]:
        """Detecta rostos numa imagem BGR (formato do OpenCV), do maior para o menor."""
        altura, largura = imagem_bgr.shape[:2]
        if (largura, altura) != self._tamanho:
            self._detector.setInputSize((largura, altura))
            self._tamanho = (largura, altura)

        _, saida = self._detector.detect(imagem_bgr)
        if saida is None:
            return []

        # Cada linha: x, y, w, h, 5 pares (x, y) de pontos, confiança.
        deteccoes = [
            Deteccao(
                caixa=linha[0:4].copy(),
                pontos=linha[4:14].reshape(5, 2).copy(),
                confianca=float(linha[14]),
            )
            for linha in saida
        ]
        deteccoes.sort(key=lambda d: d.area, reverse=True)
        return deteccoes

    def maior_rosto(self, imagem_bgr: np.ndarray) -> Deteccao | None:
        """Atalho para o controle de acesso: considera só a pessoa mais próxima da câmera."""
        deteccoes = self.detectar(imagem_bgr)
        return deteccoes[0] if deteccoes else None


def espelhar(deteccoes: list[Deteccao], largura: int) -> list[Deteccao]:
    """Coordenadas das detecções numa imagem espelhada horizontalmente (só para exibir)."""
    espelhadas = []
    for d in deteccoes:
        caixa = d.caixa.copy()
        caixa[0] = largura - d.caixa[0] - d.caixa[2]
        pontos = d.pontos.copy()
        pontos[:, 0] = largura - d.pontos[:, 0]
        espelhadas.append(Deteccao(caixa=caixa, pontos=pontos, confianca=d.confianca))
    return espelhadas


def encontrar_cameras(max_indice: int = 6) -> list[int]:
    """Índices de câmera que devolvem imagem de verdade.

    Alguns índices abrem mas só entregam quadros pretos (nós de metadados ou
    câmera infravermelha), então lê alguns quadros e descarta os escuros.
    """
    encontradas = []
    nivel_anterior = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # índices inexistentes geram avisos
    try:
        for indice in range(max_indice):
            captura = abrir_captura(indice)
            if captura.isOpened():
                for _ in range(5):  # dá tempo para a exposição automática ajustar
                    ok, quadro = captura.read()
                    if ok and quadro.mean() > 5:
                        encontradas.append(indice)
                        break
            captura.release()
    finally:
        cv2.utils.logging.setLogLevel(nivel_anterior)
    return encontradas


# ---------------------------------------------------------------------------
# Visualização e teste rápido
# ---------------------------------------------------------------------------

_CORES_PONTOS = ((255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255))


def desenhar(
    imagem_bgr: np.ndarray, deteccoes: list[Deteccao], mostrar_pontos: bool = True
) -> np.ndarray:
    """Devolve uma cópia da imagem com caixas, pontos e confiança desenhados."""
    saida = imagem_bgr.copy()
    for d in deteccoes:
        x, y, w, h = d.caixa.astype(int)
        cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(saida, f"{d.confianca:.2f}", (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        if mostrar_pontos:
            for (px, py), cor in zip(d.pontos.astype(int), _CORES_PONTOS):
                cv2.circle(saida, (px, py), 3, cor, -1)
    return saida


def _rodar_imagem(detector: DetectorFacial, caminho: str) -> None:
    imagem = cv2.imread(caminho)
    if imagem is None:
        raise SystemExit(f"Não consegui abrir a imagem: {caminho}")
    deteccoes = detector.detectar(imagem)
    print(f"{len(deteccoes)} rosto(s) encontrado(s)")
    for i, d in enumerate(deteccoes):
        print(f"  [{i}] caixa={d.caixa.astype(int).tolist()} confiança={d.confianca:.3f}")
    cv2.imshow("Deteccao - qualquer tecla para sair", desenhar(imagem, deteccoes))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _rodar_webcam(detector: DetectorFacial, camera: int) -> None:
    try:
        captura = CameraAoVivo(camera)
    except RuntimeError:
        raise SystemExit(f"Não consegui abrir a webcam {camera}. Tente --camera 1, 2...")

    anterior = time.perf_counter()
    fps = 0.0
    ultimo = 0
    try:
        while True:
            numero, quadro, _ = captura.ler()
            if numero == ultimo:          # nenhum quadro novo ainda
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            ultimo = numero
            quadro = cv2.flip(quadro, 1)  # espelha, como um espelho
            deteccoes = detector.detectar(quadro)

            agora = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(agora - anterior, 1e-6))
            anterior = agora

            tela = desenhar(quadro, deteccoes)
            cv2.putText(tela, f"{fps:4.1f} FPS | {len(deteccoes)} rosto(s) | q para sair",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Deteccao facial (YuNet)", tela)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        captura.fechar()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste do detector facial YuNet.")
    parser.add_argument("--camera", type=int, default=None,
                        help="índice da webcam (padrão: a primeira que funcionar)")
    parser.add_argument("--imagem", help="detecta numa imagem em vez da webcam")
    parser.add_argument("--confianca", type=float, default=0.9, help="confiança mínima (padrão: 0.9)")
    args = parser.parse_args()

    detector = DetectorFacial(confianca_minima=args.confianca)
    if args.imagem:
        _rodar_imagem(detector, args.imagem)
    else:
        camera = args.camera
        if camera is None:
            cameras = encontrar_cameras()
            if not cameras:
                raise SystemExit("Nenhuma webcam com imagem encontrada.")
            camera = cameras[0]
            print(f"Câmeras com imagem: {cameras}. Usando a {camera}.")
        _rodar_webcam(detector, camera)


if __name__ == "__main__":
    main()
