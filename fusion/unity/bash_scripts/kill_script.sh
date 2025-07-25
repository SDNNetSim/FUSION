#!/bin/bash

# This script terminates all running instances of a given script.
# It is intended to be used on Unix-like systems (Linux, macOS).

# Usage: ./kill_script.sh <script_path>
# Example: ./kill_script.sh /path/to/script.py

# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <script_path>"
  exit 1
fi

# Assign argument to variable for better readability
script_path="$1"
script_name=$(basename "$script_path")

# Check if the provided script path is valid
if [ ! -f "$script_path" ]; then
  echo "Error: $script_path is not a valid file path."
  exit 1
fi

# Print a message indicating the script is being killed
echo "$script_name has been killed."

# Kill all processes matching the script name using pkill
pkill -f "$script_name"
