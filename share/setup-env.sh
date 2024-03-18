#!/bin/bash

# Get the current directory where the script is located
SCRIPT_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BIN_PATH=$(realpath "$SCRIPT_PATH/../bin/")

# Check if the file is executable, if not, make it executable
if [ ! -x "$BIN_PATH/agi" ]; then
    chmod +x "$BIN_PATH/agi"
fi

# Add the directory containing agi to PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    export PATH="$PATH:$BIN_PATH"
    echo "Added $BIN_PATH to PATH."
fi

echo "Setup complete."