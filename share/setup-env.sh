#!/bin/bash
# Get the current directory where the script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_PATH=$(realpath $SCRIPT_DIR/../src/AGI.py)

# Function to set up the alias
setup_alias() {
    alias agi="$1 $SCRIPT_PATH"
    echo "Alias 'agi' created."
}

# Check for python3 and python, then set up the alias accordingly
if command -v python3 &>/dev/null; then
    setup_alias "python3"
elif command -v python &>/dev/null; then
    setup_alias "python"
else
    echo "Python is not installed or not available in PATH."
fi
