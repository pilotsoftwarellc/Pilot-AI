@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts\export_pilot11_gguf.py --hf-dir runs\pilot_1_1_llama_chat_fast\hf_last --outfile exports\pilot_1_1-llama-chat-q8_0.gguf --outtype q8_0 --write-modelfile
pause
