#!/bin/bash
# Copyright: © 2026 Avik Md Emtiaz Arefin, Muhammed Humam Hossain
# Authors: Avik Md Emtiaz Arefin, Muhammed Humam Hossain

# --- CONFIGURATION ---
ENV_NAME="darkbotx"
# ---------------------

OS=$(uname -s | sed 's/Darwin/MacOSX/')
ARCH=$(uname -m)

# 1. Ensure UV is installed
if command -v uv &> /dev/null; then
    echo "✅ [UV] is already installed: $(uv --version)"
else
    echo "🚀 [Installing UV]"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    
    if command -v uv &> /dev/null; then
        echo "✨ [UV] Installed successfully!"
    else
        echo "❌ UV installation failed. Please check your connection."
        exit 1
    fi
fi

# 2. Raspberry Pi Specific Logic
echo "🔍 Checking hardware compatibility..."

# Check if the device tree model contains "Raspberry Pi"
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "🍓 [Raspberry Pi Detected] Installing hardware dependencies..."
    
    # Update and install system library
    sudo apt update && sudo apt install -y liblgpio-dev
    
    # Add the python library via uv
    # Note: This assumes you are in a directory with a pyproject.toml
    if [ -f "pyproject.toml" ]; then
        uv add rpi-lgpio
        echo "✅ liblgpio-dev and rpi-lgpio installed."
    else
        echo "⚠️  rpi-lgpio not added to uv: No pyproject.toml found in this directory."
    fi
else
    echo "ℹ️  Not a Raspberry Pi. Skipping liblgpio-dev and rpi-lgpio installation."
fi

echo "🏁 Setup check complete."