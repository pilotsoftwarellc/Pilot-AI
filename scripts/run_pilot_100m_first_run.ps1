$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "runs\pilot_100m_10k_first_run"
$LogFile = Join-Path $LogDir "train.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $Root
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Pilot 1.0 - first 100M run, 10k context"
Write-Host "Workspace: $Root"
Write-Host "Log: $LogFile"
Write-Host ""

& $Python train.py --config configs\pilot_100m_10k_first_run.json 2>&1 | Tee-Object -FilePath $LogFile

Write-Host ""
Write-Host "Training finished. Checkpoints are in runs\pilot_100m_10k_first_run"
Read-Host "Press Enter to close"
