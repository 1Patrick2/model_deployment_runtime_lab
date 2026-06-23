#!/bin/bash
# Model Deployment Runtime Lab - WSL RKNN Setup
# Sets up RKNN Toolkit2 and dependencies for optional ONNX → RKNN conversion.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Model Deployment Runtime Lab - WSL RKNN Setup              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# WSL path detection
if [[ "$SCRIPT_DIR" == /mnt/* ]]; then
    echo -e "${YELLOW}⚠️  Detected Windows path. Using WSL filesystem for faster IO.${NC}"
    RKNN_WORKDIR="${RKNN_WORKDIR:-$HOME/mdrl-rknn-workdir}"
    echo -e "${CYAN}   Workdir: ${RKNN_WORKDIR}${NC}"
    mkdir -p "$RKNN_WORKDIR"
    cd "$RKNN_WORKDIR"
else
    RKNN_WORKDIR="${RKNN_WORKDIR:-$SCRIPT_DIR}"
fi

echo -e "${CYAN}Checking system dependencies...${NC}"
for cmd in python3 git wget unzip; do
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${RED}ERROR: $cmd not found. Install it first.${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✅ System dependencies ok${NC}"

# Virtual environment
ENV_DIR="${RKNN_WORKDIR}/rknn-env"
if [ -d "$ENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Environment already exists at ${ENV_DIR}${NC}"
else
    echo -e "${CYAN}Creating Python virtual environment...${NC}"
    python3 -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
echo -e "${GREEN}✅ Virtual environment active${NC}"

# Base requirements
REQ_FILE="${PROJECT_ROOT}/requirements_wsl_rknn.txt"
if [ -f "$REQ_FILE" ]; then
    echo -e "${CYAN}Installing base dependencies...${NC}"
    pip install -r "$REQ_FILE"
    echo -e "${GREEN}✅ Base dependencies installed${NC}"
fi

# RKNN Toolkit2
RKNN_DIR="${RKNN_WORKDIR}/rknn-toolkit2"
if [ -d "$RKNN_DIR/.git" ]; then
    echo -e "${CYAN}Updating RKNN Toolkit2...${NC}"
    cd "$RKNN_DIR"
    git pull origin main 2>/dev/null || true
else
    echo -e "${CYAN}Cloning RKNN Toolkit2 (sparse)...${NC}"
    git clone --filter=blob:none --sparse https://github.com/airockchip/rknn-toolkit2.git "$RKNN_DIR"
    cd "$RKNN_DIR"
    git sparse-checkout set packages/x86_64
fi

cd "${RKNN_DIR}/packages/x86_64" || { echo -e "${RED}packages/x86_64 not found${NC}"; exit 1; }

# Install RKNN toolkit wheel
WHEEL=$(ls -t rknn_toolkit2*cp310*.whl 2>/dev/null | head -1)
if [ -n "$WHEEL" ]; then
    echo -e "${CYAN}Installing ${WHEEL}...${NC}"
    pip install "$WHEEL"
    echo -e "${GREEN}✅ RKNN Toolkit2 installed${NC}"
else
    echo -e "${YELLOW}⚠️  No cp310 RKNN wheel found. Check ${RKNN_DIR}/packages/x86_64/ manually.${NC}"
fi

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  WSL RKNN Setup Complete!                                   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  source ${ENV_DIR}/bin/activate"
echo ""
echo -e "${YELLOW}Future:${NC}"
echo -e "  python -m src.export.convert_rknn --config configs/rknn.yaml"
