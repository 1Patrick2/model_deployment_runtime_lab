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

# ── RKNN Toolkit2 ──────────────────────────────────────────
# Use direct wheel download instead of git sparse clone to
# avoid GitHub network instability.

RKNN_WHEEL_DIR="${RKNN_WORKDIR}/rknn-wheel"
mkdir -p "$RKNN_WHEEL_DIR"
cd "$RKNN_WHEEL_DIR"

RKNN_WHEEL_URL="https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
RKNN_WHEEL_FILE="rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

if [ -f "$RKNN_WHEEL_FILE" ]; then
    echo -e "${CYAN}RKNN wheel already downloaded.${NC}"
else
    echo -e "${CYAN}Downloading RKNN Toolkit2 wheel (38 MB)...${NC}"
    wget -c -O "$RKNN_WHEEL_FILE" "$RKNN_WHEEL_URL" || {
        echo -e "${RED}Failed to download RKNN wheel. Check network or GitHub access.${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ RKNN wheel downloaded.${NC}"
fi

# Install CPU PyTorch first (avoids pulling GPU/CUDA packages)
echo -e "${CYAN}Installing CPU PyTorch 2.4.0...${NC}"
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.4.0+cpu"
echo -e "${GREEN}✅ CPU PyTorch installed.${NC}"

# Install RKNN dependencies (scipy, opencv-headless, etc.)
echo -e "${CYAN}Installing RKNN additional dependencies...${NC}"
python -m pip install --no-cache-dir \
    "protobuf==4.25.4" \
    "psutil>=5.9.0" \
    "ruamel.yaml>=0.17.21" \
    "scipy>=1.9.3" \
    "opencv-python-headless==4.10.0.84" \
    "fast-histogram>=0.11" \
    "packaging" \
    "coloredlogs" \
    "flatbuffers"
echo -e "${GREEN}✅ RKNN dependencies installed.${NC}"

# Install RKNN wheel with --no-deps so pip doesn't upgrade torch/onnx
echo -e "${CYAN}Installing RKNN Toolkit2 (--no-deps)...${NC}"
python -m pip install --no-cache-dir --no-deps "$RKNN_WHEEL_FILE"
echo -e "${GREEN}✅ RKNN Toolkit2 installed.${NC}"

# Verify
echo -e "${CYAN}Verifying RKNN Toolkit2 import...${NC}"
python -c "from rknn.api import RKNN; print('RKNN Toolkit2 OK')"

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  WSL RKNN Setup Complete!                                   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  source ${ENV_DIR}/bin/activate"
echo ""
echo -e "${YELLOW}RKNN conversion:${NC}"
echo -e "  python -m src.rknn.convert --config configs/rknn_convert.yaml"
