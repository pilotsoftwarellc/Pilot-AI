$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "runs\data_prepare_large"
$LogFile = Join-Path $LogDir "prepare.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $Root
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Pilot 1.0 large token preparation"
Write-Host "Input: data\raw"
Write-Host "Output: data\tokens"
Write-Host "Validation fraction: 0.5%"
Write-Host "Log: $LogFile"
Write-Host ""

& $Python scripts\prepare_data.py --input-dir data\raw --out-dir data\tokens --split-mode document --val-fraction 0.005 --min-val-tokens 5000000 2>&1 |
    Tee-Object -FilePath $LogFile

Write-Host ""
Write-Host "Token preparation finished. Tokens are in data\tokens"
Read-Host "Press Enter to close"
