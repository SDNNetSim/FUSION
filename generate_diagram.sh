#!/bin/bash
cd "$(git rev-parse --show-toplevel)" || exit 1
chmod +x .git/hooks/pre-commit

# Try to find pyreverse in current PATH
PYREVERSE=$(command -v pyreverse)
echo "Using Python from: $(which python)"
echo "Python version: $(python --version)"
source ./venv/Scripts/activate

echo "Running pyreverse before commit..."

MODULES=("tests" "src")  
OUTPUT_DIR="class_diagram_output"

rm -rf ${OUTPUT_DIR}/*

for MODULE in "${MODULES[@]}"; do
    echo "Generating diagrams for module: $MODULE"
    if pyreverse -ASmy "$MODULE" -o png -p "${MODULE}" -d "$OUTPUT_DIR"; then
        CLASS_FILE="${OUTPUT_DIR}/${MODULE}_classes.puml"
        PACKAGE_FILE="${OUTPUT_DIR}/${MODULE}_packages.puml"

        if [ -f "$CLASS_FILE" ]; then
            echo "Class diagram saved: $CLASS_FILE"
        else
            echo "Warning: expected class diagram not found: $CLASS_FILE"
        fi

        if [ -f "$PACKAGE_FILE" ]; then
            echo "Package diagram saved: $PACKAGE_FILE"
        else
            echo "Warning: expected package diagram not found: $PACKAGE_FILE"
        fi
    else
        echo "Warning: pyreverse failed for module $MODULE."
    fi
done

exit 0