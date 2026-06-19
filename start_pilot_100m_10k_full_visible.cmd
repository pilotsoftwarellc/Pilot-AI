@echo off
setlocal
cd /d "%~dp0"

set PYTHONUTF8=1

echo Pilot 1.0 full training
echo Model: 100M params
echo Context: 10240 tokens
echo Data: data\tokens
echo Config: configs\pilot_100m_10k.json
echo.
echo Progress will show tokens/s, average tokens/s, percent, elapsed time, ETA, and tokens processed.
echo.

set RESUME=
for /f "delims=" %%F in ('dir /b /o-d runs\pilot_100m_10k\ckpt_*.pt 2^>nul') do (
    set RESUME=runs\pilot_100m_10k\%%F
    goto found_resume
)

if exist runs\pilot_100m_10k_first_run\ckpt_last.pt (
    set RESUME=runs\pilot_100m_10k_first_run\ckpt_last.pt
)

:found_resume
if "%RESUME%"=="" (
    echo Resume: none, starting from scratch
    ".venv\Scripts\python.exe" train.py --config configs\pilot_100m_10k.json
) else (
    echo Resume: %RESUME%
    ".venv\Scripts\python.exe" train.py --config configs\pilot_100m_10k.json --resume "%RESUME%"
)

echo.
echo Full training finished. Checkpoints are in runs\pilot_100m_10k
pause
