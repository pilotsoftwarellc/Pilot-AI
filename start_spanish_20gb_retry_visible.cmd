@echo off
setlocal
cd /d "%~dp0"

set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo Pilot 1.0 Spanish corpus retry
echo Target: 20 GB Spanish from FineWeb2 spa_Latn
echo English file will not be touched.
echo Log: runs\data_download_spanish_retry\download.log
echo.

".venv\Scripts\python.exe" scripts\download_bilingual_corpus.py --skip-english --es-gb 20 --out-dir data\raw --overwrite --log-file runs\data_download_spanish_retry\download.log

echo.
echo Spanish download finished. Raw files are in data\raw
echo Next step: powershell.exe -ExecutionPolicy Bypass -File .\scripts\prepare_large_tokens.ps1
pause
