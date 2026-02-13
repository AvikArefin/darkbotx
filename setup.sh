#!/bin/bash
# Copyright: © 2026 Avik Md Emtiaz Arefin, Muhammed Humam Hossain
# Authors: Avik Md Emtiaz Arefin, Muhammed Humam Hossain

# --- CONFIGURATION (Change these to rename your setup) ---
ENV_NAME="darkbotx"
# ---------------------------------------------------------

OS=$(uname -s | sed 's/Darwin/MacOSX/')
ARCH=$(uname -m)

# Check if uv is already installed
if command -v uv &> /dev/null; then
    echo "✅ [UV] is already installed: $(uv --version)"
else
    echo "🚀 [Installing UV]"
    
    # Official installation command for Linux/macOS
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the cargo env or update PATH for the current session
    # Note: uv typically installs to ~/.local/bin
    export PATH="$HOME/.local/bin:$PATH"
    
    if command -v uv &> /dev/null; then
        echo "✨ [UV] Installed successfully!"
    else
        echo "❌ Installation failed or PATH not updated. Please restart your terminal."
    fi
fi