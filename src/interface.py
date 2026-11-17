"""Interface gráfica: detecção ao vivo e captura de fotos para o cadastro.

Uso (a partir da raiz do projeto):
    python -m src.interface

Atalho: barra de espaço captura uma foto (quando o cursor não está num campo de texto).
"""

from __future__ import annotations

import time
import tkinter as tk

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from src.alinhamento import alinhar, angulo_olhos
from src.camera import CameraAoVivo
from src.captura import LARGURA_MINIMA_ROSTO, SessaoCaptura, sessao_de_hoje
from src.deteccao import Deteccao, DetectorFacial, encontrar_cameras, espelhar

INTERVALO_MS = 5               # com que frequência verifica se chegou quadro novo
INTERVALO_SEQUENCIA_MS = 300   # pausa entre fotos na captura em sequência
FOTOS_POR_SEQUENCIA = 20

# Paleta (BGR para o OpenCV, hex para a interface)
COR = {
    "fundo": "#0E1117",
    "cartao": "#161B22",
    "borda": "#262C36",
    "texto": "#E6EDF3",
    "suave": "#8B949E",
    "destaque": "#4F8CFF",
    "destaque_hover": "#3A74E0",
    "ok": "#2EA043",
    "aviso": "#D29922",
    "erro": "#F85149",
}
BGR_DESTAQUE = (255, 140, 79)
BGR_AVISO = (34, 153, 210)
BGR_PONTOS = (255, 255, 255)


def desenhar_cantos(img: np.ndarray, d: Deteccao, cor: tuple[int, int, int], mostrar_pontos: bool) -> None:
    """Desenha a caixa só com os cantos (visual mais limpo que o retângulo inteiro)."""
    x, y, w, h = d.caixa.astype(int)
    t = max(12, min(w, h) // 5)
    for (cx, cy, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * t, cy), cor, 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * t), cor, 3, cv2.LINE_AA)
    if mostrar_pontos:
        for px, py in d.pontos.astype(int):
            cv2.circle(img, (px, py), 3, BGR_PONTOS, -1, cv2.LINE_AA)


class Cartao(ctk.CTkFrame):
    """Bloco da barra lateral com título."""

    def __init__(self, master, titulo: str) -> None:
        super().__init__(master, fg_color=COR["cartao"], corner_radius=14,
                         border_width=1, border_color=COR["borda"])
        ctk.CTkLabel(self, text=titulo.upper(), text_color=COR["suave"],
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=16, pady=(14, 6))
        self.corpo = ctk.CTkFrame(self, fg_color="transparent")
        self.corpo.pack(fill="x", padx=16, pady=(0, 14))


class Metrica(ctk.CTkFrame):
    """Pequeno indicador numérico abaixo do vídeo."""

    def __init__(self, master, rotulo: str) -> None:
        super().__init__(master, fg_color=COR["cartao"], corner_radius=10,
                         border_width=1, border_color=COR["borda"])
        ctk.CTkLabel(self, text=rotulo, text_color=COR["suave"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(8, 0))
        self.valor = ctk.CTkLabel(self, text="–", text_color=COR["texto"],
                                  font=ctk.CTkFont(size=18, weight="bold"))
        self.valor.pack(anchor="w", padx=12, pady=(0, 8))

    def definir(self, texto: str) -> None:
        self.valor.configure(text=texto)


class AppCaptura(ctk.CTk):
    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        super().__init__(fg_color=COR["fundo"])
        self.title("Controle de Acesso Facial — AM2")
        self.geometry("1240x760")
        self.minsize(1000, 640)
        self.protocol("WM_DELETE_WINDOW", self.fechar)

        self.detector = DetectorFacial(confianca_minima=0.9)
        self.captura: CameraAoVivo | None = None
        self.ultimo_quadro = 0
        self.quadro_original: np.ndarray | None = None   # sem espelho, usado na captura
        self.rosto_atual: Deteccao | None = None
        self.n_rostos = 0
        self.fps = 0.0
        self.atraso_ms = 0.0
        self.instante_anterior = time.perf_counter()
        self.restantes_sequencia = 0

        self.cameras = encontrar_cameras()
        self.var_espelhar = tk.BooleanVar(value=True)
        self.var_pontos = tk.BooleanVar(value=True)

        self._montar_layout()
        self.bind("<space>", self._atalho_espaco)
        if self.cameras:
            self.abrir_camera(self.cameras[0])
        else:
            self._mostrar_estado("Nenhuma câmera encontrada", COR["erro"])
        self._loop()

    # ------------------------------------------------------------------ layout
    def _montar_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- coluna do vídeo
        esquerda = ctk.CTkFrame(self, fg_color="transparent")
        esquerda.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        esquerda.grid_rowconfigure(1, weight=1)
        esquerda.grid_columnconfigure(0, weight=1)

        topo = ctk.CTkFrame(esquerda, fg_color="transparent")
        topo.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(topo, text="Controle de Acesso Facial", text_color=COR["texto"],
                     font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")
        self.estado = ctk.CTkLabel(topo, text="  Iniciando…  ", corner_radius=999,
                                   fg_color=COR["borda"], text_color=COR["texto"],
                                   font=ctk.CTkFont(size=12, weight="bold"), height=28)
        self.estado.pack(side="right")

        self.moldura_video = ctk.CTkFrame(esquerda, fg_color="#000000", corner_radius=16,
                                          border_width=1, border_color=COR["borda"])
        self.moldura_video.grid(row=1, column=0, sticky="nsew")
        self.video = ctk.CTkLabel(self.moldura_video, text="")
        self.video.place(relx=0.5, rely=0.5, anchor="center")

        metricas = ctk.CTkFrame(esquerda, fg_color="transparent")
        metricas.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.metricas = {}
        for i, nome in enumerate(("FPS", "Atraso", "Rostos", "Confiança", "Inclinação")):
            metricas.grid_columnconfigure(i, weight=1)
            m = Metrica(metricas, nome)
            m.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            self.metricas[nome] = m

        # ---- barra lateral
        lateral = ctk.CTkFrame(self, fg_color="transparent", width=330)
        lateral.grid(row=0, column=1, sticky="ns", padx=(10, 20), pady=20)

        cam = Cartao(lateral, "Câmera")
        cam.pack(fill="x", pady=(0, 12))
        opcoes = [f"Câmera {i}" for i in self.cameras] or ["Nenhuma"]
        self.menu_camera = ctk.CTkOptionMenu(
            cam.corpo, values=opcoes, width=290, fg_color=COR["borda"],
            button_color=COR["borda"], button_hover_color=COR["destaque"],
            command=lambda v: self.abrir_camera(int(v.split()[-1])) if v != "Nenhuma" else None)
        self.menu_camera.pack(fill="x", pady=(0, 10))
        ctk.CTkSwitch(cam.corpo, text="Espelhar imagem", variable=self.var_espelhar,
                      progress_color=COR["destaque"]).pack(anchor="w", pady=3)
        ctk.CTkSwitch(cam.corpo, text="Mostrar pontos do rosto", variable=self.var_pontos,
                      progress_color=COR["destaque"]).pack(anchor="w", pady=3)

        det = Cartao(lateral, "Detecção")
        det.pack(fill="x", pady=(0, 12))
        linha = ctk.CTkFrame(det.corpo, fg_color="transparent")
        linha.pack(fill="x")
        ctk.CTkLabel(linha, text="Confiança mínima", text_color=COR["texto"]).pack(side="left")
        self.rotulo_confianca = ctk.CTkLabel(linha, text="0.90", text_color=COR["destaque"],
                                             font=ctk.CTkFont(weight="bold"))
        self.rotulo_confianca.pack(side="right")
        self.slider = ctk.CTkSlider(det.corpo, from_=0.5, to=0.99, number_of_steps=49,
                                    progress_color=COR["destaque"], button_color=COR["destaque"],
                                    button_hover_color=COR["destaque_hover"],
                                    command=self._mudar_confianca)
        self.slider.set(0.9)
        self.slider.pack(fill="x", pady=(6, 0))

        cad = Cartao(lateral, "Cadastro")
        cad.pack(fill="x")
        ctk.CTkLabel(cad.corpo, text="Nome", text_color=COR["suave"]).pack(anchor="w")
        self.entrada_nome = ctk.CTkEntry(cad.corpo, placeholder_text="ex.: Gustavo Gimenes",
                                         border_color=COR["borda"], fg_color=COR["fundo"])
        self.entrada_nome.pack(fill="x", pady=(2, 8))
        ctk.CTkLabel(cad.corpo, text="Sessão", text_color=COR["suave"]).pack(anchor="w")
        self.entrada_sessao = ctk.CTkEntry(cad.corpo, border_color=COR["borda"], fg_color=COR["fundo"])
        self.entrada_sessao.insert(0, sessao_de_hoje())
        self.entrada_sessao.pack(fill="x", pady=(2, 12))
        for entrada in (self.entrada_nome, self.entrada_sessao):
            entrada.bind("<KeyRelease>", lambda _e: self._atualizar_contagem())

        previa = ctk.CTkFrame(cad.corpo, fg_color="transparent")
        previa.pack(fill="x", pady=(0, 12))
        self.previa = ctk.CTkLabel(previa, text="sem\nfoto", width=96, height=96, corner_radius=10,
                                   fg_color=COR["fundo"], text_color=COR["suave"])
        self.previa.pack(side="left")
        info = ctk.CTkFrame(previa, fg_color="transparent")
        info.pack(side="left", padx=12, fill="both", expand=True)
        self.rotulo_contagem = ctk.CTkLabel(info, text="0", text_color=COR["texto"],
                                            font=ctk.CTkFont(size=28, weight="bold"))
        self.rotulo_contagem.pack(anchor="w")
        ctk.CTkLabel(info, text="fotos nesta sessão", text_color=COR["suave"]).pack(anchor="w")

        self.botao_foto = ctk.CTkButton(cad.corpo, text="Capturar foto  (Espaço)", height=38,
                                        fg_color=COR["destaque"], hover_color=COR["destaque_hover"],
                                        font=ctk.CTkFont(weight="bold"), command=self.capturar_uma)
        self.botao_foto.pack(fill="x", pady=(0, 8))
        self.botao_sequencia = ctk.CTkButton(
            cad.corpo, text=f"Capturar {FOTOS_POR_SEQUENCIA} em sequência", height=38,
            fg_color="transparent", border_width=1, border_color=COR["destaque"],
            text_color=COR["destaque"], hover_color=COR["borda"], command=self.alternar_sequencia)
        self.botao_sequencia.pack(fill="x")
        # A barra só aparece durante a captura em sequência.
        self.progresso = ctk.CTkProgressBar(cad.corpo, progress_color=COR["destaque"], height=6)
        self.progresso.set(0)
        self.aviso = ctk.CTkLabel(cad.corpo, text="Mexa levemente a cabeça entre as fotos.",
                                  text_color=COR["suave"], wraplength=290, justify="left")
        self.aviso.pack(anchor="w", pady=(8, 0))

    # ------------------------------------------------------------------ câmera
    def abrir_camera(self, indice: int) -> None:
        if self.captura is not None:
            self.captura.fechar()
            self.captura = None
        try:
            self.captura = CameraAoVivo(indice)
            self.ultimo_quadro = 0
        except RuntimeError:
            self._mostrar_estado(f"Câmera {indice} indisponível", COR["erro"])

    def _mudar_confianca(self, valor: float) -> None:
        self.rotulo_confianca.configure(text=f"{valor:.2f}")
        self.detector.definir_confianca(valor)

    def _loop(self) -> None:
        if self.captura is not None:
            numero, quadro, chegada = self.captura.ler()
            if numero != self.ultimo_quadro and quadro is not None:
                self.ultimo_quadro = numero
                self._processar(quadro)
                atraso = (time.perf_counter() - chegada) * 1000
                self.atraso_ms = 0.9 * self.atraso_ms + 0.1 * atraso
        self.after(INTERVALO_MS, self._loop)

    def _processar(self, quadro: np.ndarray) -> None:
        # Detecta no quadro original; o espelho é aplicado só na exibição.
        deteccoes = self.detector.detectar(quadro)
        self.quadro_original = quadro
        self.rosto_atual = deteccoes[0] if deteccoes else None
        self.n_rostos = len(deteccoes)

        altura, largura = quadro.shape[:2]
        tela = quadro.copy()
        mostrar = deteccoes
        if self.var_espelhar.get():
            tela = cv2.flip(tela, 1)
            mostrar = espelhar(deteccoes, largura)
        for i, d in enumerate(mostrar):
            pequeno = d.caixa[2] < LARGURA_MINIMA_ROSTO
            cor = BGR_AVISO if (pequeno or i > 0) else BGR_DESTAQUE
            desenhar_cantos(tela, d, cor, self.var_pontos.get())

        self._exibir_video(tela)
        self._atualizar_indicadores(largura)

    def _exibir_video(self, tela_bgr: np.ndarray) -> None:
        disp_l = max(self.moldura_video.winfo_width() - 24, 320)
        disp_a = max(self.moldura_video.winfo_height() - 24, 240)
        altura, largura = tela_bgr.shape[:2]
        escala = min(disp_l / largura, disp_a / altura)
        tamanho = (int(largura * escala), int(altura * escala))
        imagem = Image.fromarray(cv2.cvtColor(tela_bgr, cv2.COLOR_BGR2RGB))
        foto = ctk.CTkImage(light_image=imagem, dark_image=imagem, size=tamanho)
        self.video.configure(image=foto)
        self.video.image = foto  # mantém a referência viva

    def _atualizar_indicadores(self, largura_quadro: int) -> None:
        agora = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(agora - self.instante_anterior, 1e-6)
        self.instante_anterior = agora

        r = self.rosto_atual
        self.metricas["FPS"].definir(f"{self.fps:.0f}")
        self.metricas["Atraso"].definir(f"{self.atraso_ms:.0f} ms")
        self.metricas["Rostos"].definir(str(self.n_rostos))
        self.metricas["Confiança"].definir(f"{r.confianca:.2f}" if r else "–")
        self.metricas["Inclinação"].definir(f"{angulo_olhos(r):+.0f}°" if r else "–")

        if self.restantes_sequencia:
            return  # durante a sequência o estado mostra o progresso
        if r is None:
            self._mostrar_estado("Nenhum rosto", COR["borda"])
        elif r.caixa[2] < LARGURA_MINIMA_ROSTO:
            self._mostrar_estado("Aproxime-se da câmera", COR["aviso"])
        elif self.n_rostos > 1:
            self._mostrar_estado("Mais de um rosto", COR["aviso"])
        else:
            self._mostrar_estado("Rosto pronto", COR["ok"])

    def _mostrar_estado(self, texto: str, cor: str) -> None:
        self.estado.configure(text=f"   {texto}   ", fg_color=cor)

    # ------------------------------------------------------------------ captura
    def _sessao(self) -> SessaoCaptura | None:
        try:
            return SessaoCaptura(self.entrada_nome.get(), self.entrada_sessao.get())
        except ValueError as erro:
            self._avisar(str(erro), COR["erro"])
            return None

    def _atualizar_contagem(self) -> None:
        try:
            sessao = SessaoCaptura(self.entrada_nome.get(), self.entrada_sessao.get())
            self.rotulo_contagem.configure(text=str(sessao.quantidade()))
        except ValueError:
            self.rotulo_contagem.configure(text="0")

    def _avisar(self, texto: str, cor: str = COR["suave"]) -> None:
        self.aviso.configure(text=texto, text_color=cor)

    def capturar_uma(self) -> bool:
        sessao = self._sessao()
        if sessao is None:
            return False
        r, quadro = self.rosto_atual, self.quadro_original
        if r is None or quadro is None:
            self._avisar("Nenhum rosto detectado.", COR["aviso"])
            return False
        if r.caixa[2] < LARGURA_MINIMA_ROSTO:
            self._avisar("Rosto muito pequeno: aproxime-se da câmera.", COR["aviso"])
            return False
        if self.n_rostos > 1:
            self._avisar("Mais de um rosto na imagem: só a pessoa cadastrada deve aparecer.", COR["aviso"])
            return False

        rosto = alinhar(quadro, r)
        caminho = sessao.salvar(quadro, rosto)
        self._mostrar_previa(rosto)
        self.rotulo_contagem.configure(text=str(sessao.quantidade()))
        self._avisar(f"Salva: {sessao.nome}/{sessao.sessao}/{caminho.name}", COR["ok"])
        return True

    def _mostrar_previa(self, rosto_bgr: np.ndarray) -> None:
        imagem = Image.fromarray(cv2.cvtColor(rosto_bgr, cv2.COLOR_BGR2RGB))
        foto = ctk.CTkImage(light_image=imagem, dark_image=imagem, size=(96, 96))
        self.previa.configure(image=foto, text="")
        self.previa.image = foto

    def alternar_sequencia(self) -> None:
        if self.restantes_sequencia:
            self._parar_sequencia("Sequência interrompida.")
            return
        if self._sessao() is None:
            return
        self.restantes_sequencia = FOTOS_POR_SEQUENCIA
        self.progresso.set(0)
        self.progresso.pack(fill="x", pady=(10, 0), before=self.aviso)
        self.botao_sequencia.configure(text="Parar sequência")
        self._passo_sequencia()

    def _passo_sequencia(self) -> None:
        if not self.restantes_sequencia:
            return
        if self.capturar_uma():
            self.restantes_sequencia -= 1
            feitas = FOTOS_POR_SEQUENCIA - self.restantes_sequencia
            self.progresso.set(feitas / FOTOS_POR_SEQUENCIA)
            self._mostrar_estado(f"Capturando {feitas}/{FOTOS_POR_SEQUENCIA}", COR["destaque"])
        if self.restantes_sequencia:
            self.after(INTERVALO_SEQUENCIA_MS, self._passo_sequencia)
        else:
            self._parar_sequencia(f"{FOTOS_POR_SEQUENCIA} fotos salvas.")

    def _parar_sequencia(self, mensagem: str) -> None:
        self.restantes_sequencia = 0
        self.progresso.pack_forget()
        self.botao_sequencia.configure(text=f"Capturar {FOTOS_POR_SEQUENCIA} em sequência")
        self._avisar(mensagem, COR["ok"])

    def _atalho_espaco(self, evento) -> None:
        # Não captura enquanto a pessoa digita nos campos de texto.
        if isinstance(evento.widget, (tk.Entry, ctk.CTkEntry)):
            return
        self.capturar_uma()

    def fechar(self) -> None:
        self.restantes_sequencia = 0
        if self.captura is not None:
            self.captura.fechar()
        self.destroy()


def main() -> None:
    AppCaptura().mainloop()


if __name__ == "__main__":
    main()
