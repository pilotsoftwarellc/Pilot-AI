@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts\setup_pilot11_vision.py --config configs\pilot_1_1_vision_siglip.json
pause
