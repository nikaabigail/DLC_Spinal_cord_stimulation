<#
.SYNOPSIS
    Thin wrapper that finds a suitable Python and runs check_environment.py.

.DESCRIPTION
    Prefers the project venv (C:\dlc_live_env\Scripts\python.exe) because the
    installed-package checks are only meaningful inside that interpreter.
    Falls back to any python on PATH so the probe still runs on a half-set-up
    machine (it will just WARN that it is not the venv).

    Usage:
        powershell -ExecutionPolicy Bypass -File _migration\check_environment.ps1
#>

$ErrorActionPreference = 'Stop'

# Resolve the python script next to this wrapper.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$pyScript  = Join-Path $scriptDir 'check_environment.py'

if (-not (Test-Path $pyScript)) {
    Write-Host "ERROR: check_environment.py not found next to this wrapper ($pyScript)" -ForegroundColor Red
    exit 2
}

# Candidate interpreters, in order of preference.
$candidates = @(
    'C:\dlc_live_env\Scripts\python.exe',                 # project venv (best)
    "$env:DLC_LIVE_PYTHON"                                # optional override
)

$python = $null
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) { $python = $c; break }
}

# Fall back to whatever python / py is on PATH.
if (-not $python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source }
}
if (-not $python) {
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source }
}

if (-not $python) {
    Write-Host "ERROR: no Python interpreter found. Install Python 3.10 and/or create C:\dlc_live_env." -ForegroundColor Red
    exit 2
}

Write-Host "Using Python: $python" -ForegroundColor Cyan
Write-Host "Running     : $pyScript" -ForegroundColor Cyan
Write-Host ("-" * 74)

& $python $pyScript
exit $LASTEXITCODE
