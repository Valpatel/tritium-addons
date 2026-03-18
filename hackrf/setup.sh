#!/bin/bash
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
# HackRF Addon Setup — installs all required tools and dependencies.
#
# Usage: ./setup.sh
#
# This script installs:
# 1. hackrf tools (hackrf_info, hackrf_sweep, hackrf_transfer, hackrf_spiflash)
# 2. rtl_433 (ISM band device decoder — TPMS, weather stations, etc)
# 3. Python dependencies (numpy, scipy for signal processing)
# 4. udev rules for HackRF permissions

set -e

echo "=== HackRF Addon Setup ==="
echo ""

# 1. HackRF tools
echo "[1/4] HackRF tools..."
if command -v hackrf_info &>/dev/null; then
    echo "  Already installed: $(hackrf_info --version 2>&1 | head -1)"
else
    echo "  Installing hackrf..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq hackrf libhackrf-dev
fi

# 2. rtl_433
echo "[2/4] rtl_433..."
if command -v rtl_433 &>/dev/null; then
    echo "  Already installed: $(rtl_433 -V 2>&1 | head -1)"
else
    echo "  Installing rtl_433..."
    sudo apt-get install -y -qq rtl-433 || {
        echo "  rtl_433 not in apt, trying snap..."
        sudo snap install rtl-433 2>/dev/null || {
            echo "  Installing from source..."
            cd /tmp
            git clone https://github.com/merbanan/rtl_433.git
            cd rtl_433 && mkdir build && cd build
            cmake .. && make -j$(nproc) && sudo make install
            cd / && rm -rf /tmp/rtl_433
        }
    }
fi

# 3. Python dependencies
echo "[3/4] Python dependencies..."
DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$DIR/.venv/bin/pip"
if [ -f "$VENV" ]; then
    $VENV install numpy scipy 2>/dev/null && echo "  numpy + scipy installed" || echo "  Already installed"
else
    echo "  WARNING: venv not found at $DIR/.venv"
fi

# 4. udev rules
echo "[4/4] HackRF udev rules..."
RULES_FILE="/etc/udev/rules.d/52-hackrf.rules"
if [ -f "$RULES_FILE" ]; then
    echo "  Already configured"
else
    echo 'ATTR{idVendor}=="1d50", ATTR{idProduct}=="6089", MODE="0666", GROUP="plugdev"' | sudo tee "$RULES_FILE" > /dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "  udev rules installed"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Verify:"
echo "  hackrf_info           # Device detection"
echo "  hackrf_sweep -f 88:108 -w 500000 -1  # Quick FM sweep"
echo "  rtl_433 -f 315M -M level  # TPMS monitoring"
