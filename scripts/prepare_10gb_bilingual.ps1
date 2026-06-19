$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

Set-Location -LiteralPath $Root
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

& $Python scripts\download_bilingual_corpus.py --en-gb 5 --es-gb 5 --out-dir data\raw --overwrite
& $Python scripts\prepare_data.py --input-dir data\raw --out-dir data\tokens --min-val-tokens 20000
