from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


CHAT_TEMPLATE = """{% if messages[0]['role'] != 'system' %}Sistema: Eres Pilot 0.0.1, un asistente bilingue de IA. Responde claro, util y directo.
{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}Sistema: {{ message['content'] }}
{% elif message['role'] == 'user' %}Usuario: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Pilot: {{ message['content'] }}{{ eos_token }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Pilot:{% endif %}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tokenizer", default="gpt2")
    parser.add_argument("--out-dir", default="tokenizers/pilot_0_0_1_gpt2_chat")
    parser.add_argument("--model-max-length", type=int, default=4096)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.model_max_length
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(out_dir)

    metadata = {
        "base_tokenizer": args.base_tokenizer,
        "vocab_size": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "model_max_length": args.model_max_length,
        "note": "Tokenizer only. Pilot 0.0.1 model weights are trained from scratch.",
    }
    (out_dir / "pilot_tokenizer_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved tokenizer to {out_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
