# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PORT          ?= 8000

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean serve

clean:
	rm -rf $(BUILDDIR)

serve: html
	@echo "Documentation available at: http://localhost:$(PORT)"
	@echo "Press Ctrl+C to stop"
	cd $(BUILDDIR)/html && python -m http.server $(PORT)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
