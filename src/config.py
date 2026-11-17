"""Caminhos e constantes compartilhados pelo projeto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS_DIR = RAIZ / "dados"
MODELOS_DIR = RAIZ / "modelos"
IMAGENS_DIR = RAIZ / "imagens"
BUILD_DIR = RAIZ / "build"

# Detector facial YuNet (OpenCV Zoo, licença MIT).
YUNET_ARQUIVO = MODELOS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
