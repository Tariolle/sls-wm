TEX = pdflatex -interaction=nonstopmode
BIB = bibtex
SRC = main

all: $(SRC).pdf

$(SRC).pdf: $(SRC).tex references.bib checklist.tex neurips_2026.sty
	$(TEX) $(SRC)
	$(BIB) $(SRC)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f $(SRC).{aux,bbl,blg,log,out,pdf,synctex.gz,fls,fdb_latexmk}

.PHONY: all clean
