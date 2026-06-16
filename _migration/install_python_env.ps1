<#
================================================================================
 install_python_env.ps1
 Recreate the LEAN DLC live-inference virtual environment on a clean Windows PC.

 Target: Windows 11/10 x64, desktop RTX 5070 (Blackwell sm_120), recent
         Blackwell-capable NVIDIA driver already installed.

 What it does:
   1. Verifies Python 3.10.x is available (3.10.11 expected).
   2. Creates a fresh venv at $VenvDir (default C:\dlc_live_env).
   3. Upgrades pip.
   4. Installs torch/torchvision/torchaudio +cu128 from the PyTorch CUDA 12.8
      wheel index (these are NOT on PyPI).
   5. Installs the rest of the pinned set from PyPI via requirements.txt.
   6. Smoke-tests: torch.cuda + dlclive import.

 It does NOT install: the Daheng Galaxy SDK (gxipy), the NVIDIA driver, the
 model files, or the full 'deeplabcut' training package. See the migration guide.

 Usage (from an elevated-or-normal PowerShell):
   .\install_python_env.ps1
   .\install_python_env.ps1 -VenvDir C:\dlc_live_env -PythonExe "C:\Path\to\python.exe"
================================================================================
#>

[CmdletBinding()]
param(
    [string]$VenvDir  = "C:\dlc_live_env",
    # Path to the Python 3.10 base interpreter. If empty, the script tries the
    # py launcher (py -3.10) and then a bare 'python' on PATH.
    [string]$PythonExe = "",
    [string]$ReqFile  = (Join-Path $PSScriptRoot "requirements.txt")
)

$ErrorActionPreference = "Stop"
$CU128_INDEX = "https://download.pytorch.org/whl/cu128"

function Resolve-BasePython {
    param([string]$Explicit)
    if ($Explicit -and (Test-Path $Explicit)) { return $Explicit }
    # Prefer the py launcher pinned to 3.10
    $py = (Get-Command py -ErrorAction SilentlyContinue)
    if ($py) {
        $v = & py -3.10 -c "import sys;print('%d.%d.%d'%sys.version_info[:3])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { return "py -3.10" }
    }
    $p = (Get-Command python -ErrorAction SilentlyContinue)
    if ($p) { return $p.Source }
    throw "No Python 3.10 found. Install Python 3.10.11 (64-bit) first: https://www.python.org/downloads/release/python-31011/"
}

function Invoke-BasePython {
    param([string]$Spec, [string[]]$ArgList)
    if ($Spec -eq "py -3.10") { & py -3.10 @ArgList } else { & $Spec @ArgList }
}

Write-Host "==> Resolving base Python 3.10 ..." -ForegroundColor Cyan
$base = Resolve-BasePython -Explicit $PythonExe
$ver  = (Invoke-BasePython $base @("-c","import sys;print('%d.%d.%d'%sys.version_info[:3])")).Trim()
Write-Host "    base interpreter: $base  (Python $ver)"
if ($ver -notmatch '^3\.10\.') {
    throw "Base Python is $ver but 3.10.x is required (source machine: 3.10.11). Aborting."
}
if ($ver -ne "3.10.11") {
    Write-Warning "Base Python is $ver, source machine used 3.10.11. Any 3.10.x will work but 3.10.11 is the exact match."
}

if (Test-Path $VenvDir) {
    throw "Venv dir '$VenvDir' already exists. Remove it first (Remove-Item -Recurse -Force '$VenvDir') or pass -VenvDir."
}

Write-Host "==> Creating venv at $VenvDir ..." -ForegroundColor Cyan
Invoke-BasePython $base @("-m","venv",$VenvDir)

$VPy = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VPy)) { throw "venv python not found at $VPy" }

Write-Host "==> Upgrading pip ..." -ForegroundColor Cyan
& $VPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Host "==> Installing torch/torchvision/torchaudio (+cu128) from $CU128_INDEX ..." -ForegroundColor Cyan
& $VPy -m pip install `
    "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128" `
    --index-url $CU128_INDEX
if ($LASTEXITCODE -ne 0) { throw "torch +cu128 install failed (check network / index URL)" }

Write-Host "==> Installing the rest from PyPI ($ReqFile) ..." -ForegroundColor Cyan
if (-not (Test-Path $ReqFile)) { throw "requirements.txt not found at $ReqFile" }
& $VPy -m pip install -r $ReqFile
if ($LASTEXITCODE -ne 0) { throw "requirements.txt install failed" }

Write-Host "==> pip check (informational) ..." -ForegroundColor Cyan
# NOTE: this will print ONE expected warning:
#   'deeplabcut-live requires opencv-python-headless, which is not installed'
# That is HARMLESS — we install opencv-python (full) instead, which provides the
# same cv2 module the live bridge needs. Do not 'fix' it by adding the headless
# build alongside (two cv2 providers can clash). Matches the source machine.
& $VPy -m pip check

Write-Host "==> Smoke test: torch CUDA + dlclive import ..." -ForegroundColor Cyan
& $VPy -c @"
import torch, dlclive, cv2, numpy
print('torch        :', torch.__version__)
print('cuda runtime :', torch.version.cuda)
print('cuda avail   :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device       :', torch.cuda.get_device_name(0))
    print('capability   :', torch.cuda.get_device_capability(0), '(expect (12, 0) = sm_120 Blackwell)')
print('dlclive      :', dlclive.__version__)
print('cv2          :', cv2.__version__)
print('numpy        :', numpy.__version__)
"@
if ($LASTEXITCODE -ne 0) { throw "smoke test failed" }

Write-Host ""
Write-Host "DONE. Lean live-inference env ready at $VenvDir" -ForegroundColor Green
Write-Host "Remaining manual steps (NOT done by this script):" -ForegroundColor Yellow
Write-Host "  * Install the Daheng Galaxy SDK (provides gxipy via Samples/Python SDK)."
Write-Host "  * Copy the model project (config.yaml + exported-models-pytorch snapshot)."
Write-Host "  * Copy the camera config from C:\config_daheng\."
Write-Host "  * If torch.cuda.is_available() is False -> update the NVIDIA driver (Blackwell)."
