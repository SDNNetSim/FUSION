#!/bin/bash
# scripts/analyze_dependencies.sh

echo "📊 Generating dependency analysis..."

# Create output directory
mkdir -p reports/analysis

# Create dependency graphs
echo "🔗 Module dependencies..."
pydeps fusion --show-deps --max-bacon=3 -o reports/analysis/dependencies.png

# Find circular dependencies
echo "🔄 Checking for circular dependencies..."
pydeps fusion --show-cycles > reports/analysis/circular_dependencies.txt

# Dead code detection
echo "🧹 Finding dead code..."
vulture fusion/ .vulture_whitelist.py --min-confidence=80 > reports/analysis/dead_code.txt

# Generate detailed dependency report
echo "📋 Generating dependency report..."
pydeps fusion --show-deps --max-bacon=2 --cluster > reports/analysis/dependency_report.txt

echo "✅ Analysis complete! Check reports/analysis/ for results"
echo "📁 Generated files in reports/analysis/:"
echo "  - dependencies.png: Visual dependency graph"
echo "  - circular_dependencies.txt: Circular dependency report"
echo "  - dead_code.txt: Dead code analysis"
echo "  - dependency_report.txt: Detailed dependency report"
