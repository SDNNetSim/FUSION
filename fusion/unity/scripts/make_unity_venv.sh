#!/bin/bash

# Unity virtual environment creation script for FUSION cluster computing.
# Creates Python virtual environments for Unity-managed SLURM jobs with proper
# validation and error handling.
#
# Usage: ./make_unity_venv.sh <target_directory> <python_version>
# Example: ./make_unity_venv.sh /work/venvs/unity_env python3.11

set -e

readonly SCRIPT_NAME="$(basename "$0")"

validate_python_version() {
    local python_version="$1"

    if ! command -v "$python_version" &>/dev/null; then
        echo "❌ Error: Python version '$python_version' not found" >&2
        echo "Available Python versions:" >&2
        command -v python3 &>/dev/null && python3 --version >&2
        command -v python &>/dev/null && python --version >&2
        return 1
    fi

    echo "✅ Found Python version: $("$python_version" --version)"
    return 0
}

create_unity_virtual_environment() {
    local target_directory="$1"
    local python_version="$2"

    # Create target directory if it doesn't exist
    if [[ ! -d "$target_directory" ]]; then
        echo "🔧 Creating target directory: $target_directory"
        mkdir -p "$target_directory"
    fi

    # Change to target directory
    cd "$target_directory" || {
        echo "❌ Error: Cannot access directory '$target_directory'" >&2
        exit 1
    }

    echo "📂 Working in directory: $(pwd)"

    # Remove existing venv if present
    if [[ -d "venv" ]]; then
        echo "🗑️  Removing existing virtual environment"
        rm -rf venv
    fi

    # Create virtual environment
    echo "🔧 Creating virtual environment with $python_version"
    "$python_version" -m venv venv

    # Verify virtual environment creation
    if [[ -f "venv/bin/activate" ]]; then
        echo "✅ Virtual environment 'venv' created successfully in '$target_directory'"
        echo "📝 To activate: source $target_directory/venv/bin/activate"
    else
        echo "❌ Error: Virtual environment creation failed" >&2
        exit 1
    fi
}

main() {
    # Check argument count
    if [[ $# -ne 2 ]]; then
        echo "Usage: $SCRIPT_NAME <target_directory> <python_version>" >&2
        echo "Example: $SCRIPT_NAME /work/venvs/unity_env python3.11" >&2
        exit 1
    fi

    local target_directory="$1"
    local python_version="$2"

    echo "🌟 Starting Unity virtual environment creation"
    echo "Target directory: $target_directory"
    echo "Python version: $python_version"

    # Validate inputs
    validate_python_version "$python_version" || exit 1

    # Create virtual environment
    create_unity_virtual_environment "$target_directory" "$python_version"

    echo "🎉 Unity virtual environment setup completed successfully!"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
