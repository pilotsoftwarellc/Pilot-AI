$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "runs\data_download_120gb"
$LogFile = Join-Path $LogDir "download.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $Root
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Pilot 1.0 bilingual corpus download"
Write-Host "Target: 120 GB raw JSONL"
Write-Host "English: 60 GB from FineWeb-Edu sample-10BT"
Write-Host "Spanish: 60 GB from FineWeb2 spa_Latn"
Write-Host "Log: $LogFile"
Write-Host ""

& $Python scripts\download_bilingual_corpus.py --en-gb 60 --es-gb 60 --out-dir data\raw --overwrite 2>&1 |
    Tee-Object -FilePath $LogFile

Write-Host ""
Write-Host "Download finished. Raw files are in data\raw"
Write-Host "Next step: .\scripts\prepare_large_tokens.ps1"
Read-Host "Press Enter to close"
