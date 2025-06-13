#!/bin/bash

set -e  # Fail on any error

# Navigate to repo root
SCRIPT_DIR=$(dirname "$0")
REPO_DIR=$(realpath "$SCRIPT_DIR/..")
cd "$REPO_DIR"

# Debugging: Log current directory
echo "Current working directory: $(pwd)"

# Check pylint installation
if ! command -v pylint &> /dev/null; then
  echo "pylint not found. Install it with 'pip install pylint'."
  exit 1
fi

# Identify Python files, excluding specific files or directories
echo "Identifying Python files..."

IGNORE_DIRS=(
  "./bash"
  "./.venv"
  "./docs"
)

# Add paths to ignore using -path and -prune (ONLY ONCE).
# Build find command with dynamic pruning
FIND_CMD=(find .)

# Start group for pruning
FIND_CMD+=('(')
for dir in "${IGNORE_DIRS[@]}"; do
  FIND_CMD+=(-path "$dir" -o)
done
unset 'FIND_CMD[-1]'  # Remove the trailing -o
FIND_CMD+=(')' -prune -o -name "*.py" -print)

# Run find and capture results
PYTHON_FILES=$("${FIND_CMD[@]}")

# Debugging: Log all found Python files
echo "Found Python files:"
echo "$PYTHON_FILES"

# Run pylint on each file
echo "Running pylint..."
for file in $PYTHON_FILES; do
  echo "Linting $file"
  pylint "$file" || exit 1
done

echo "Linting completed successfully!"
