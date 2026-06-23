# Model Deployment Runtime Lab - Windows Setup Script
# 此脚本用于配置模型部署运行时的 Conda 环境

$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
if ([string]::IsNullOrEmpty($ScriptDir)) { $ScriptDir = (Get-Location).Path }

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  Model Deployment Runtime Lab - Windows Setup                ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host "`nThis script will set up Conda environments for:" -ForegroundColor Yellow
Write-Host "  1. mdrl-runtime — ONNX Runtime, ZMQ, benchmark"
Write-Host "  2. mdrl-dev     — pytest, dev tools"
Write-Host "  3. mdrl-train   — optional: PyTorch baseline, pruning (with --with-train)`n"

function Fail($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

function CheckCommand($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Fail "$name not found. $hint"
    }
}

Write-Host "Checking required commands..." -ForegroundColor Cyan
CheckCommand conda "Please install Miniconda or Anaconda: https://docs.conda.io/en/latest/miniconda.html"
CheckCommand git "Please install Git (https://git-scm.com/)"
Write-Host "✅ Required commands present`n" -ForegroundColor Green

# Accept Conda Terms of Service
Write-Host "Accepting Conda Terms of Service..." -ForegroundColor Cyan
& conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>$null
& conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>$null

function EnsureCondaEnv($name, $reqFile) {
    Write-Host "`n>>> Environment: $name" -ForegroundColor Cyan
    $envs = & conda env list --json | ConvertFrom-Json
    $exists = $envs.envs | Where-Object { $_ -match "\\$name$" }
    if ($exists) {
        Write-Host "  ✅ Conda environment '$name' already exists. Updating dependencies..." -ForegroundColor Yellow
        if ($reqFile -and (Test-Path $reqFile)) {
            & conda run -n $name pip install -r $reqFile 2>&1 | ForEach-Object { Write-Host "     $_" }
            if ($LASTEXITCODE -ne 0) { Fail "pip install failed for $name in $reqFile" }
        }
        Write-Host "  ✅ $name up to date`n" -ForegroundColor Green
    } else {
        Write-Host "  Creating Conda environment '$name' (python=3.10)..." -ForegroundColor Yellow
        & conda create -n $name python=3.10 -y 2>&1 | ForEach-Object { Write-Host "     $_" }
        if ($LASTEXITCODE -ne 0) { Fail "conda create failed for $name" }
        if ($reqFile -and (Test-Path $reqFile)) {
            & conda run -n $name pip install -r $reqFile 2>&1 | ForEach-Object { Write-Host "     $_" }
            if ($LASTEXITCODE -ne 0) { Fail "pip install failed for $name in $reqFile" }
        }
        Write-Host "  ✅ $name created and dependencies installed`n" -ForegroundColor Green
    }
}

# Parse optional flags
$withTrain = $args -contains '--with-train'

# 1. mdrl-runtime
EnsureCondaEnv "mdrl-runtime" (Join-Path $ScriptDir "requirements_win_runtime.txt")

# 2. mdrl-dev
EnsureCondaEnv "mdrl-dev" (Join-Path $ScriptDir "requirements_win_dev.txt")

# 3. Optional: mdrl-train
if ($withTrain) {
    EnsureCondaEnv "mdrl-train" (Join-Path $ScriptDir "requirements_win_train.txt")
} else {
    Write-Host "`n>>> Environment: mdrl-train (skipped, use --with-train to create)" -ForegroundColor Cyan
}

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  Setup Complete!                                            ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  conda activate mdrl-runtime" -ForegroundColor Cyan
Write-Host "  python verify_paths.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Future stages:" -ForegroundColor Yellow
Write-Host "  python -m pytest tests -q                        # Stage 1+ tests"
Write-Host "  python -m src.server.zmq_server --backend fake   # Stage 1: ZMQ server"
Write-Host ""
Write-Host "Note: PyTorch is NOT auto-installed. If you need training:" -ForegroundColor Magenta
Write-Host "  1. Activate mdrl-train (conda activate mdrl-train)" -ForegroundColor Magenta
Write-Host "  2. Install PyTorch from https://pytorch.org (CUDA or CPU)" -ForegroundColor Magenta
