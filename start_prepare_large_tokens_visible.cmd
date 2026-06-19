@echo off
setlocal
cd /d "%~dp0"

set PYTHONUTF8=1

echo Pilot 1.0 token preparation
echo Input: data\raw
echo Output: data\tokens
echo Split: document-balanced validation
echo Log: runs\data_prepare_large\prepare.log
echo.

if not exist runs\data_prepare_large mkdir runs\data_prepare_large

".venv\Scripts\python.exe" scripts\prepare_data.py --input-dir data\raw --out-dir data\tokens --split-mode document --val-fraction 0.005 --min-val-tokens 5000000 > runs\data_prepare_large\prepare.log 2>&1

type runs\data_prepare_large\prepare.log
echo.
echo Token preparation finished. Tokens are in data\tokens
pause
