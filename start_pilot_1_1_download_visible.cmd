@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts\download_bilingual_corpus.py --en-gb 1 --es-gb 1 --out-dir data\raw --overwrite
pause
