# =====================
# Configuration
# =====================
FILE   = rapport
LATEX  = pdflatex -shell-escape -interaction=nonstopmode -halt-on-error -file-line-error
BIBTEX = biber

# =====================
# Règle par défaut : force la recompilation
# =====================
all: force

# =====================
# Compilation forcée
# =====================
force:
	@echo "=== Removing old PDF"
	@rm -f $(FILE).pdf
	@echo "=== LaTeX Compilation (1/3)"
	@$(LATEX) $(FILE).tex > /dev/null
	@echo "=== Biber"
	@$(BIBTEX) $(FILE) > /dev/null
	@echo "=== LaTeX Compilation (2/3)"
	@$(LATEX) $(FILE).tex > /dev/null
	@echo "=== LaTeX Compilation (3/3)"
	@$(LATEX) $(FILE).tex > /dev/null
	@echo "=== Cleaning temporary files"
	@rm -f *.aux *.bbl *.bcf *.blg *.log *.nlo *.toc *.out *.lof *.lot *.fls *run.xml *.fdb_latexmk *.synctex.gz
	@echo "=== $(FILE).pdf generated"

.PHONY: all force
