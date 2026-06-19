@echo off
setlocal
cd /d "%~dp0"

set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo Pilot 1.0 bilingual corpus download
echo Target: 120 GB raw JSONL
echo English: 60 GB from FineWeb-Edu sample-10BT
echo Spanish: 60 GB from FineWeb2 spa_Latn
echo Log: runs\data_download_120gb\download.log
echo.

".venv\Scripts\python.exe" scripts\download_bilingual_corpus.py --en-gb 60 --es-gb 60 --out-dir data\raw --overwrite --log-file runs\data_download_120gb\download.log

echo.
echo Download finished. Raw files are in data\raw
echo Next step: scripts\prepare_large_tokens.ps1
pause
