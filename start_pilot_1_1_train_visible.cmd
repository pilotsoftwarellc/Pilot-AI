@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if exist runs\pilot_1_1_llama_chat_fast\hf_last\trainer_state.pt (
  python scripts\train_pilot11_llama.py --config configs\pilot_1_1_llama_chat_fast.json --resume runs\pilot_1_1_llama_chat_fast\hf_last
) else (
  python scripts\train_pilot11_llama.py --config configs\pilot_1_1_llama_chat_fast.json
)
pause
