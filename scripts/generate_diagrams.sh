#!/bin/bash
# scripts/generate_diagrams.sh

echo "📈 Generating project diagrams..."

# Create output directory
mkdir -p reports/diagrams

# UML class diagrams with pyreverse
echo "🏗️  Generating UML diagrams..."
pyreverse -o png -p fusion fusion/ --output-dir reports/diagrams/

# Dependency graphs with pydeps
echo "🔗 Generating dependency graphs..."
pydeps fusion --show-deps --max-bacon=3 -o reports/diagrams/dependencies.png

# Module-specific diagrams
for module in core utils cli modules interfaces configs; do
    if [ -d "fusion/$module" ]; then
        echo "📊 Generating diagram for fusion.$module..."
        pydeps fusion.$module -o reports/diagrams/deps_$module.png
    fi
done

# Generate architecture overview
echo "🎯 Generating architecture overview..."
pydeps fusion --show-deps --max-bacon=2 --cluster -o reports/diagrams/architecture_overview.png

# Generate module interaction diagram
echo "🔄 Generating module interactions..."
pydeps fusion --show-deps --max-bacon=1 -o reports/diagrams/module_interactions.png

echo "✅ Diagrams generated in reports/diagrams/"
echo "📁 Generated files:"
echo "  - classes_fusion.png: UML class diagram"
echo "  - packages_fusion.png: UML package diagram"
echo "  - dependencies.png: Overall dependency graph"
echo "  - architecture_overview.png: High-level architecture"
echo "  - module_interactions.png: Module interaction diagram"
echo "  - deps_*.png: Individual module dependency graphs"
