from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


OLLAMA_TEMPLATE = '''FROM ./pilot_1_1-llama-chat-q8_0.gguf

PARAMETER num_ctx 4096
PARAMETER temperature 0.8
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.08
PARAMETER stop "<|endoftext|>"

TEMPLATE """Sistema: Eres Pilot 1.1, un asistente bilingue de IA. Responde claro, util y directo.
{{ if .Prompt }}Usuario: {{ .Prompt }}
{{ end }}Pilot:"""
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", default="runs/pilot_1_1_llama_chat_fast/hf_last")
    parser.add_argument("--outfile", default="exports/pilot_1_1-llama-chat-q8_0.gguf")
    parser.add_argument("--outtype", default="q8_0", choices=["f16", "q8_0"])
    parser.add_argument("--write-modelfile", action="store_true")
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    if not hf_dir.exists():
        raise SystemExit(f"HF checkpoint not found: {hf_dir}")
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ".venv\\Scripts\\python.exe",
        "external\\llama.cpp\\convert_hf_to_gguf.py",
        str(hf_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        args.outtype,
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {outfile}")

    if args.write_modelfile:
        modelfile = outfile.parent / "Modelfile.pilot-1.1"
        modelfile.write_text(OLLAMA_TEMPLATE, encoding="utf-8")
        print(f"wrote {modelfile}")


if __name__ == "__main__":
    main()
