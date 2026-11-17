# Plano de trabalho

> Documento interno do grupo, temporário: será removido antes da entrega final. A descrição do projeto e as instruções de uso ficam no [README](README.md).

O eixo técnico é **transfer learning para reconhecimento facial em conjunto aberto com poucas amostras**: uma CNN gera um vetor (embedding) para cada rosto, e uma regra de decisão libera só quem estiver perto o bastante de um integrante cadastrado. Cadastrar alguém novo não exige retreinar a rede, basta guardar os vetores.

## Prazo e regras do enunciado

- **Entrega: 30/11 às 14h no AVA** — artigo/relatório **+ vídeo + slides** (faltar qualquer um penaliza a nota).
- **Apresentação ao vivo** no Google Meet: 15 min (±5 min); passar do tempo penaliza.
- Nota final = nota da apresentação (individual, 0 a 1) × nota do relatório (grupo, 0 a 10): todos precisam dominar o conteúdo.
- O enunciado não exige template; mantemos o SBC já montado em `artigo/`.
- Temas do enunciado cobertos: **redes convolucionais**, **encoders** (a CNN que gera o vetor) e **fine-tuning**.

Itens marcados como *(opcional)* só entram se sobrar tempo.

## Calendário

| Período | Foco |
|---|---|
| 17 a 20/11 | postar no AVA; embedding + cadastro + decisão na interface; LFW; **coleta das sessões** |
| 21 a 25/11 | experimentos (regras de decisão, fotos de cadastro, FAR) e fine-tuning |
| 26 a 29/11 | artigo, slides, gravação da demo e do vídeo, ensaio cronometrado |
| 30/11 (manhã) | folga para imprevistos; entrega até 14h |

## 1. Garantir o tema (hoje)
- [ ] Postar no AVA: integrantes; temas "Redes Convolucionais / Encoders + fine-tuning (transfer learning)"; dados "imagens faciais próprias + LFW". Nenhum grupo pode repetir a combinação.
- [ ] Termo de consentimento assinado pelos 4 para uso das fotos (LGPD).

## 2. Pipeline base (rede congelada) — até 20/11
- [x] Detecção facial com YuNet (`src/deteccao.py`)
- [x] Interface ao vivo com baixa latência (`src/interface.py`, `src/camera.py`)
- [x] Alinhamento pelos 5 pontos e recorte 112×112 no padrão ArcFace (`src/alinhamento.py`)
- [x] Captura na interface → `dados/integrantes/<nome>/<sessão>/` (`src/captura.py`)
- [ ] Embedding com CNN pré-treinada (ArcFace/SFace)
- [ ] Cadastro + decisão por similaridade de cosseno → **LIBERADO / NEGADO** na interface

## 3. Dados — até 22/11
- [ ] Cada integrante grava 3 sessões em dias diferentes (sugestão: 18, 20 e 22/11), variando luz, óculos, distância e ângulo
- [ ] Baixar o LFW (desconhecidos) e passá-lo pelo mesmo pipeline
- [ ] Divisão fixa em arquivo: integrantes por **sessão**, desconhecidos por **identidade**

## 4. Regras de decisão (em cima da rede congelada) — 21 a 23/11
- [ ] Comparar limiar no centróide, kNN, One-Class SVM, SVM com negativos e softmax com limiar (linha de base fechada)
- [ ] Número de fotos de cadastro: 1, 3, 5, 10, 20
- [ ] Limiar calibrado para FAR de 1% e 0,1%; curvas DET e EER
- [ ] Impostores parecidos (os mais próximos de cada integrante no LFW)
- [ ] *(opcional)* Crescimento do banco de cadastrados (LFW como membros falsos: 4, 10, 50, 100)

## 5. Transfer learning — 22 a 25/11

| Nível | O que se faz | No projeto |
|---|---|---|
| 1. Extração de características | rede pré-treinada congelada | ArcFace/SFace como estão (passos 2 e 4) |
| 2. Fine-tuning parcial | descongela as últimas camadas, perda ArcFace | adaptar a rede aos dados |
| 3. Fine-tuning completo | rede inteira, taxa de aprendizado baixa | verificar se compensa ou causa overfitting |

- [ ] Níveis 1 × 2
- [ ] *(opcional)* Nível 3
- [ ] *(opcional)* Origem do conhecimento: ResNet pré-treinada no **ImageNet** × rede pré-treinada **em rostos**
- [ ] *(opcional)* Cadastro depois do fine-tuning: ajustar sem um integrante e cadastrá-lo depois, para ver se a rede se especializou demais
- Cortado pelo prazo: treino do zero (ResNet18 no CelebA), que exigiria horas de GPU e um download grande. No artigo, a comparação com "sem transferência" fica citada pela literatura.

O fine-tuning usa várias identidades públicas **mais** as fotos do grupo: pessoas do LFW com 10 ou mais fotos, **separadas** das usadas como desconhecidos no teste (CelebA só se faltarem identidades). Ajustar só com os 4 integrantes tende a piorar a rejeição de desconhecidos.

## 6. Entrega — 26 a 30/11
- [ ] Figuras e tabelas geradas por script em `imagens/`
- [ ] Artigo/relatório (`artigo/main.tex`)
- [ ] Slides (`apresentacao/slides.tex`) — também são entregues
- [ ] Demo final com votação entre quadros, gravada para o vídeo
- [ ] Vídeo da apresentação (15 min ±5), com todos falando
- [ ] Ensaio cronometrado da apresentação ao vivo
- [ ] Subir artigo + vídeo + slides no AVA até **30/11 às 14h**

## Divisão sugerida

| Pessoa | Frente |
|---|---|
| A | pipeline (alinhamento, captura, interface, demo) |
| B | dados (coleta, LFW, divisões) e avaliação (métricas, curvas) |
| C | regras de decisão e experimentos do passo 4 |
| D | transfer learning (passo 5) |

O artigo é escrito por todos, cada um cuidando da seção da sua frente.

