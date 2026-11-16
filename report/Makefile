MAIN   = artigo
SRC    = src
OUT    = build
PDF    = $(OUT)/$(MAIN).pdf

# MacTeX binaries, even in shells that lack /Library/TeX/texbin on PATH.
TEXBIN = PATH="/Library/TeX/texbin:$$PATH"

# latexmk runs inside src/ (-cd) so referencias.bib, secoes/, tabelas/ and figuras/ resolve;
# all aux files and the PDF go to build/.
LATEXMK = $(TEXBIN) latexmk -pdf -cd -interaction=nonstopmode -halt-on-error -outdir=../$(OUT)

SOURCES = $(wildcard $(SRC)/*.tex $(SRC)/*.bib $(SRC)/secoes/*.tex $(SRC)/tabelas/*.tex $(SRC)/figuras/*)

.PHONY: all figures watch check pages clean

all: $(PDF)

$(PDF): $(SOURCES)
	$(LATEXMK) $(SRC)/$(MAIN).tex

# Figures and table bodies from ../results/*.json and dados/exemplo.json (needs uv, not TeX).
figures:
	cd .. && uv run --group explain python report/scripts/figuras.py

# Rebuilds on every save.
watch:
	$(LATEXMK) -pvc -view=none $(SRC)/$(MAIN).tex

# Fails on undefined references or citations; lists overfull boxes.
check: $(PDF)
	@! grep -E "LaTeX Warning: (Reference|Citation).*undefined|There were undefined" $(OUT)/$(MAIN).log
	@grep -c "Overfull" $(OUT)/$(MAIN).log | sed 's/^/overfull boxes: /'

pages: $(PDF)
	@pdfinfo $(PDF) | awk '/^Pages:/{print "Páginas: " $$2}'

clean:
	rm -rf $(OUT)/*
