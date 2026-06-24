# Model Deployment Runtime Lab - Windows Setup Script
# ASCII-only version to avoid PowerShell encoding issues.

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
if ([string]::IsNullOrEmpty($ScriptDir)) {
    $ScriptDir = (Get-Location).Path
}

Write-Host "============================================================"
Write-Host " Model Deployment Runtime Lab - Windows Setup"
Write-Host "============================================================"
Write-Host ""
Write-Host "This script will set up Conda environments:"
Write-Host "  1. mdrl-runtime : ONNX Runtime, ZMQ, benchmark"
Write-Host "  2. mdrl-dev     : pytest and dev tools"
Write-Host "  3. mdrl-train   : optional PyTorch baseline and pruning"
Write-Host ""

function Fail($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

function CheckCommand($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Fail "$name not found. $hint"
    }
}

function InstallRequirements($envName, $reqFile) {
    if (-not $reqFile) {
        return
    }

    if (-not (Test-Path $reqFile -PathType Leaf)) {
        Fail "Requirements file not found: $reqFile"
    }

    Write-Host "Installing requirements for $envName from $reqFile"
    & conda run -n $envName pip install -r $reqFile
    if ($LASTEXITCODE -ne 0) {
        Fail "pip install failed for $envName using $reqFile"
    }
}

function EnsureCondaEnv($name, $reqFile) {
    Write-Host ""
    Write-Host ">>> Environment: $name" -ForegroundColor Cyan

    $envsJson = & conda env list --json
    if ($LASTEXITCODE -ne 0) {
        Fail "conda env list failed"
    }

    $envs = $envsJson | ConvertFrom-Json
    $exists = $false

    foreach ($envPath in $envs.envs) {
        $leaf = Split-Path $envPath -Leaf
        if ($leaf -eq $name) {
            $exists = $true
            break
        }
    }

    if ($exists) {
        Write-Host "Conda environment '$name' already exists. Updating dependencies."
    }
    else {
        Write-Host "Creating Conda environment '$name' with python=3.10"
        & conda create -n $name python=3.10 -y
        if ($LASTEXITCODE -ne 0) {
            Fail "conda create failed for $name"
        }
    }

    InstallRequirements $name $reqFile
    Write-Host "Environment '$name' is ready." -ForegroundColor Green
}

Write-Host "Checking required commands..."
CheckCommand conda "Please install Miniconda or Miniforge."
CheckCommand git "Please install Git."
Write-Host "Required commands found."
Write-Host ""

# Conda ToS command is available in some conda versions, but not all.
# Run it best-effort only.
Write-Host "Checking Conda Terms of Service if supported..."
& conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>$null
& conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>$null
Write-Host "Conda ToS check completed or skipped."
Write-Host ""

$runtimeReq = Join-Path $ScriptDir "requirements_win_runtime.txt"
$devReq = Join-Path $ScriptDir "requirements_win_dev.txt"
$trainReq = Join-Path $ScriptDir "requirements_win_train.txt"

# mdrl-runtime is the main environment for running demos and tests.
EnsureCondaEnv "mdrl-runtime" $runtimeReq
InstallRequirements "mdrl-runtime" $devReq

# mdrl-dev is kept as a lightweight development-only environment.
EnsureCondaEnv "mdrl-dev" $devReq

if ($withTrain) {
    EnsureCondaEnv "mdrl-train" $trainReq
}
else {
    Write-Host ""
    Write-Host ">>> Environment: mdrl-train skipped."
    Write-Host 'Run ".\setup_win.ps1 --with-train" to create it.'
}

Write-Host ""
Write-Host "============================================================"
Write-Host " Setup Complete"
Write-Host "============================================================"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  conda activate mdrl-runtime"
Write-Host "  python verify_paths.py"
Write-Host "  python -m pytest tests -q"
Write-Host ""
Write-Host "Stage 1 smoke test:"
Write-Host "  Terminal 1:"
Write-Host "    python -m src.server.zmq_server --backend fake"
Write-Host "  Terminal 2:"
Write-Host "    python -m src.server.zmq_client --input samples/images/danger_scene.jpg"
Write-Host ""
Write-Host "Note:"
Write-Host "  PyTorch is not installed automatically."
Write-Host "  If needed, create train env with:"
Write-Host "    .\setup_win.ps1 --with-train"