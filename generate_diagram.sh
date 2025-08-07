#!/bin/bash
cd "$(git rev-parse --show-toplevel)" || exit 1
chmod +x .git/hooks/pre-commit

# Try to find pyreverse in current PATH
PYREVERSE=$(command -v pyreverse)
echo "Using Python from: $(which python)"
echo "Python version: $(python --version)"
source ./venv/Scripts/activate

echo "Running pyreverse before commit..."


#detect packages
USER_MODULES=()  # Leave empty to auto-detect
if [ ${#USER_MODULES[@]} -gt 0 ]; then
    echo "Using user-defined module list:"
    MODULES=("${USER_MODULES[@]}")
else
    MODULES=()
    while IFS= read -r package_dir; do
        package_dir="${package_dir#./}"
        MODULES+=("$package_dir")
    done < <(
        find . -type f -name "__init__.py" \
            -not -path "./.git/*" \
            -not -path "./venv/*" \
            -not -path "./gui_scripts/*" \
            -not -path "./__pycache__/*" \
            -exec dirname {} \; | sort -u
    )
fi
printf "Detected packages:\n"
printf " - %s\n" "${MODULES[@]}"


#set the output folder and delete anything inside it before generating
OUTPUT_DIR="class_diagram_output"
rm -rf ${OUTPUT_DIR}/*

for MODULE in "${MODULES[@]}"; do
    echo "Generating diagrams for module: $MODULE"
    if pyreverse -ASmy "$MODULE" -o png -p "${MODULE}" -d "$OUTPUT_DIR"; then
        CLASS_FILE="${OUTPUT_DIR}/${MODULE}_classes.puml"
        PACKAGE_FILE="${OUTPUT_DIR}/${MODULE}_packages.puml"

    else
        echo "Warning: pyreverse failed for module $MODULE."
    fi
done

exit 0