@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts\setup_pilot11_tokenizer.py
python scripts\prepare_pilot11_llama_data.py --out-dir data\pilot_1_1_tokens_chatmix --max-gb-per-language 0.35 --chat-repeats 500000 --max-val-tokens 1000000
pause
