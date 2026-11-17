# Controle de acesso facial em conjunto aberto

Sistema de controle de acesso por reconhecimento facial que **libera as pessoas cadastradas e rejeita qualquer outra**, inclusive quem o sistema nunca viu. Trabalho 2 da disciplina Aprendizado de Máquina 2 (UFSCar, 2026/2).

## O problema

Um classificador comum com N classes sempre escolhe uma delas: um estranho na frente da câmera seria classificado como "o cadastrado mais parecido". Controle de acesso é um problema de **conjunto aberto**: no teste aparecem pessoas que não existiam no treino, e o sistema precisa saber dizer "não conheço".

O projeto estuda esse problema com **poucas fotos de cadastro por pessoa** e usando **transfer learning**: aproveitar uma CNN já treinada para reconhecer rostos em geral e adaptá-la, em vez de treinar do zero.

## Como funciona

```
webcam → detecção → alinhamento → CNN (encoder) → vetor de 512 números → regra de decisão → LIBERADO / NEGADO
```

1. **Detecção:** o YuNet (OpenCV) encontra o rosto e 5 pontos de referência (olhos, nariz e cantos da boca).
2. **Alinhamento:** os 5 pontos são usados para girar, escalar e recortar o rosto em 112×112, no padrão que as redes de reconhecimento esperam.
3. **Encoder:** uma CNN transforma o rosto num vetor. Fotos da mesma pessoa geram vetores próximos; de pessoas diferentes, vetores distantes.
4. **Cadastro:** cada pessoa autorizada tira algumas fotos, e o sistema guarda os vetores. **Cadastrar alguém novo não exige retreinar a rede.**
5. **Decisão:** o vetor do rosto na câmera é comparado com os cadastrados. Se estiver perto o bastante de alguém, o acesso é liberado; se não, é negado.

O trabalho compara redes pré-treinadas (rede congelada × fine-tuning) e regras de decisão (limiar, kNN, One-Class SVM, SVM com negativos), medindo a taxa de falsa aceitação (estranhos liberados) e de falsa rejeição (cadastrados barrados). Os desconhecidos de teste vêm do dataset público [LFW](https://vis-www.cs.umass.edu/lfw/).

## Estado atual

| Etapa | Situação |
|---|---|
| Detecção facial (YuNet) | ✅ pronta |
| Alinhamento 112×112 | ✅ pronto |
| Interface ao vivo com captura de fotos | ✅ pronta |
| Encoder (CNN pré-treinada) e cadastro | 🚧 em andamento |
| Decisão LIBERADO / NEGADO | 🚧 em andamento |
| Avaliação e fine-tuning | ⏳ a fazer |

## Instalação

Requer **Python 3.12**. Em Linux com GPU NVIDIA, o PyTorch instalado pelo pip já vem com suporte a CUDA. Sem GPU, tudo roda na CPU, só que o treino fica mais lento.

```bash
git clone git@github.com:guugimeness/open-set-face-access.git
cd open-set-face-access
python3 -m venv .venv
source .venv/bin/activate          # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para conferir se a GPU foi reconhecida:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

No Linux, a interface usa o Tkinter do sistema. Se der erro de `tkinter` ao abrir, instale com `sudo apt install python3-tk`.

## Como rodar

Sempre a partir da raiz do projeto, com o ambiente ativado. Na primeira execução, o modelo YuNet (cerca de 230 KB) é baixado automaticamente para `modelos/`.

### Interface

```bash
python -m src.interface
```

Abre uma janela com o vídeo da webcam e a detecção ao vivo. Na barra lateral:

- **Câmera:** escolha a webcam (a interface lista só as que entregam imagem), espelhe a imagem e mostre ou esconda os pontos do rosto.
- **Detecção:** ajuste a confiança mínima do detector.
- **Cadastro:** capture as fotos de uma pessoa (veja abaixo).

Abaixo do vídeo aparecem FPS, atraso, número de rostos, confiança e inclinação da cabeça.

### Capturar as fotos de cadastro

1. Na interface, digite o nome da pessoa e confira a sessão (o padrão é a data de hoje).
2. Fique sozinho na imagem e perto da câmera, até o indicador no topo mostrar **Rosto pronto**.
3. Clique em **Capturar 20 em sequência** e mexa levemente a cabeça (ângulo, expressão) durante a captura. A **barra de espaço** tira uma foto avulsa.
4. Repita em **dias diferentes**, variando luz, óculos e distância. A avaliação separa as fotos por sessão, então mais sessões dão resultados mais confiáveis.

A foto só é salva com exatamente um rosto na imagem e com o rosto grande o bastante. Cada foto gera dois arquivos:

```
dados/integrantes/<nome>/<sessão>/alinhadas/0001.png   rosto alinhado 112×112 (entrada da CNN)
dados/integrantes/<nome>/<sessão>/quadros/0001.jpg     quadro original (para refazer o alinhamento, se preciso)
```

### Detecção pela linha de comando

```bash
python -m src.deteccao                    # primeira webcam com imagem (q para sair)
python -m src.deteccao --camera 2         # webcam específica
python -m src.deteccao --imagem foto.jpg  # detecta numa imagem
```

## Estrutura

```
├── src/
│   ├── interface.py     interface gráfica (vídeo ao vivo + captura)
│   ├── deteccao.py      detector facial YuNet
│   ├── alinhamento.py   alinhamento e recorte 112×112
│   ├── captura.py       salvamento das fotos de cadastro
│   ├── camera.py        leitura da webcam com atraso mínimo
│   └── config.py        caminhos do projeto
├── artigo/              artigo (LaTeX, template SBC)
├── apresentacao/        slides (LaTeX Beamer)
├── dados/               fotos e datasets — locais, fora do git
├── modelos/             pesos baixados ou treinados — fora do git
├── imagens/             figuras do artigo e dos slides
└── build/               PDFs e arquivos compilados — fora do git
```

## Artigo e slides

O artigo e os slides são compilados com `latexmk`. Todos os arquivos gerados vão para `build/`:

```bash
make          # artigo e slides
make artigo   # build/artigo/main.pdf
make slides   # build/apresentacao/slides.pdf
make clean    # limpa build/
```

## Privacidade

Fotos de rosto são dados biométricos, considerados dados pessoais sensíveis pela LGPD. As fotos dos integrantes são coletadas com consentimento, ficam só na máquina local e **nunca são versionadas**: `dados/` e `modelos/` estão no `.gitignore`. Os datasets públicos usados (LFW) seguem as licenças de uso para pesquisa.

## Autores

Gustavo Cesar Bento Laurindo, Gustavo de Oliveira Gimenes, Lucas Vasconcelos Fujii e Mairon Cesar de Paula Faria — Departamento de Computação, UFSCar.
