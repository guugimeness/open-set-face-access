# Compila os documentos LaTeX; todos os arquivos gerados ficam em build/.
#   make            -> artigo e slides
#   make artigo     -> build/artigo/main.pdf
#   make slides     -> build/apresentacao/slides.pdf
#   make clean      -> apaga build/

LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error -cd

.PHONY: all artigo slides clean

all: artigo slides

artigo:
	$(LATEXMK) -outdir=../build/artigo artigo/main.tex

slides:
	$(LATEXMK) -outdir=../build/apresentacao apresentacao/slides.tex

clean:
	rm -rf build/*
