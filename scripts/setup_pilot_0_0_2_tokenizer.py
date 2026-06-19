from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot_lm.chat_corpus import EOS_TOKEN, PAD_TOKEN, chat_texts
from pilot_lm.text_cleaning import clean_training_text


def iter_jsonl_text(path: Path, max_bytes: int | None) -> Iterable[str]:
    consumed = 0
    with path.open("rb") as f:
        for raw_line in f:
            consumed += len(raw_line)
            if max_bytes is not None and consumed > max_bytes:
                break
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if isinstance(text, str):
                text = clean_training_text(text)
                if len(text) >= 80:
                    yield text


def training_iterator(args) -> Iterable[str]:
    for text in chat_texts():
        yield text
    max_bytes = int(args.max_gb_per_language * (1024**3)) if args.max_gb_per_language else None
    for path_str in (args.english, args.spanish):
        path = Path(path_str)
        if path.exists():
            yield from iter_jsonl_text(path, max_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tokenizers/pilot_0_0_2_bpe16k")
    parser.add_argument("--english", default="data/raw/en_fineweb_edu.jsonl")
    parser.add_argument("--spanish", default="data/raw/es_fineweb2_spa_latn.jsonl")
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-gb-per-language", type=float, default=0.25)
    parser.add_argument("--model-max-length", type=int, default=4096)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        training_iterator(args),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[EOS_TOKEN, PAD_TOKEN],
        show_progress=True,
    )
    tokenizer.save_model(str(out_dir))

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        bos_token=EOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        model_max_length=args.model_max_length,
    )
    wrapped.chat_template = """{% if messages[0]['role'] != 'system' %}Sistema: Eres Pilot 0.0.2, un asistente bilingue de IA. Responde claro, util y directo.
{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}Sistema: {{ message['content'] }}
{% elif message['role'] == 'user' %}Usuario: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Pilot: {{ message['content'] }}{{ eos_token }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Pilot:{% endif %}"""
    wrapped.save_pretrained(out_dir)

    metadata = {
        "version": "0.0.2",
        "tokenizer": "byte-level-bpe",
        "vocab_size": len(wrapped),
        "target_vocab_size": args.vocab_size,
        "eos_token": wrapped.eos_token,
        "eos_token_id": wrapped.eos_token_id,
        "pad_token": wrapped.pad_token,
        "pad_token_id": wrapped.pad_token_id,
        "max_gb_per_language": args.max_gb_per_language,
        "note": "Tokenizer trained from local English/Spanish data plus Pilot chat examples.",
    }
    (out_dir / "pilot_tokenizer_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
