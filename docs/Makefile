# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PORT          ?= 8000

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Custom FUSION targets:"
	@echo "  clean-all      Remove all build files and generated content"
	@echo "  serve          Build docs and serve them locally at http://localhost:$(PORT)"
	@echo "  autogen        Run the API documentation auto-generator"
	@echo "  autogen-clean  Clean and regenerate API documentation"

.PHONY: help Makefile clean-all serve autogen autogen-clean

# Clean everything including generated files
clean-all:
	@echo "Removing all build files and generated content..."
	rm -rf $(BUILDDIR)
	rm -rf api/_autosummary
	@echo "Clean complete!"

# Build and serve documentation locally
serve: html
	@echo "========================================"
	@echo "Starting documentation server..."
	@echo "========================================"
	@echo ""
	@echo "Documentation available at:"
	@echo "  http://localhost:$(PORT)"
	@echo ""
	@echo "Press Ctrl+C to stop the server"
	@echo ""
	cd $(BUILDDIR)/html && python -m http.server $(PORT)

# Run API documentation auto-generator
autogen:
	@echo "Running API documentation auto-generator..."
	python autogen.py --verbose

# Clean and regenerate API documentation
autogen-clean:
	@echo "Cleaning and regenerating API documentation..."
	python autogen.py --clean --verbose

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
