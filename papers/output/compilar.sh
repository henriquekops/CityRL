#!/bin/sh
# Gera artigo.pdf a partir de artigo.tex (XeLaTeX + BibTeX) e apaga os arquivos auxiliares.
# Uso: ./compilar.sh          compila
#      ./compilar.sh abrir    compila e abre o PDF
set -e
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode artigo.tex >/dev/null
bibtex artigo >/dev/null
xelatex -interaction=nonstopmode artigo.tex >/dev/null
xelatex -interaction=nonstopmode artigo.tex | grep -E "^!|Output written" || true
rm -f artigo.aux artigo.bbl artigo.blg artigo.log artigo.out
[ "$1" = "abrir" ] && open artigo.pdf
exit 0
