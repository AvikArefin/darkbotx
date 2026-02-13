#!/bin/bash
# Copyright: © 2026 Avik Md Emtiaz Arefin, Muhammed Humam Hossain
# Authors: Avik Md Emtiaz Arefin, Muhammed Humam Hossain

# Check if uv is already installed
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "✅ uv is already installed: $(uv --version)" -ForegroundColor Green
} else {
    Write-Host "🚀 uv not found. Starting installation..." -ForegroundColor Cyan
    
    # Official installation command for Windows
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    # Refresh the Path variable for the current session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host "✨ uv installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Installation finished, but 'uv' is not in the PATH yet. You may need to restart PowerShell." -ForegroundColor Yellow
    }
}