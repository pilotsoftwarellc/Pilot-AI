@echo off
setlocal
cd /d "%~dp0"

set PYTHONUTF8=1

echo Pilot 1.0 first training run
echo Model: 100M params
echo Context: 10240 tokens
echo Data: data\tokens
echo Config: configs\pilot_100m_10k_first_run.json
echo.

".venv\Scripts\python.exe" train.py --config configs\pilot_100m_10k_first_run.json

echo.
echo Training finished. Checkpoints are in runs\pilot_100m_10k_first_run
pause
