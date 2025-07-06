#!/bin/bash

# This script creates a Python virtual environment in a specified target directory
# using a specified version of Python. It is intended to be used on the Unity cluster
# maintained by UMass Amherst.

# Usage: make_venv.sh <target_directory> <python_version>
# Example: ./make_venv.sh /path/to/target_directory python3.8
pwd
# Check if the correct number of arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: make_venv.sh <target_directory> <python_version>"
  exit 1
fi

# Assign arguments to variables for better readability
target_directory="$1"
python_version="$2"

# Check if the specified Python version is installed and available
if ! command -v "$python_version" &>/dev/null; then
  echo "Python version '$python_version' could not be found. Please install it or choose a valid version."
  exit 1
fi

# Change to the target directory, exit if the directory does not exist
cd "$target_directory" || {
  echo "Directory '$target_directory' does not exist."
  exit 1
}

# Create a virtual environment named 'venv' using the specified Python version
"$python_version" -m venv venv

# Print a message indicating that the virtual environment was successfully created
echo "Virtual environment 'venv' created in '$target_directory' using $python_version!"
