from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot_lm.chat_corpus import chat_texts
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
            if not isinstance(text, str):
                continue
            text = clean_training_text(text)
            if len(text) >= 80:
                yield text


def write_tokens(f, tokens: list[int]) -> int:
    arr = np.asarray(tokens, dtype=np.uint16)
    arr.tofile(f)
    return int(arr.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", default="tokenizers/pilot_0_0_2_bpe16k")
    parser.add_argument("--out-dir", default="data/pilot_0_0_2_pretrain")
    parser.add_argument("--english", default="data/raw/en_fineweb_edu.jsonl")
    parser.add_argument("--spanish", default="data/raw/es_fineweb2_spa_latn.jsonl")
    parser.add_argument("--max-gb-per-language", type=float, default=2.0)
    parser.add_argument("--chat-repeats", type=int, default=20_000)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--max-val-tokens", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if len(tokenizer) >= 65535:
        raise ValueError(f"vocab size {len(tokenizer)} does not fit uint16")

    max_bytes = int(args.max_gb_per_language * (1024**3)) if args.max_gb_per_language else None
    stats = {
        "version": "0.0.2",
        "tokenizer_dir": args.tokenizer_dir,
        "vocab_size": len(tokenizer),
        "max_gb_per_language": args.max_gb_per_language,
        "chat_repeats": args.chat_repeats,
        "train_tokens": 0,
        "val_tokens": 0,
        "documents": 0,
    }

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    chats = chat_texts()
    started = time.perf_counter()
    with train_path.open("wb") as train_f, val_path.open("wb") as val_f:
        for _ in range(args.chat_repeats):
            tokens = tokenizer.encode(random.choice(chats), add_special_tokens=False)
            stats["train_tokens"] += write_tokens(train_f, tokens)

        for lang, path_str in (("en", args.english), ("es", args.spanish)):
            for text in iter_jsonl_text(Path(path_str), max_bytes):
                stats["documents"] += 1
                wrapped = f"Texto ({lang}):\n{text}{tokenizer.eos_token}\n"
                tokens = tokenizer.encode(wrapped, add_special_tokens=False)
                if stats["documents"] % args.val_every == 0 and stats["val_tokens"] < args.max_val_tokens:
                    stats["val_tokens"] += write_tokens(val_f, tokens)
                else:
                    stats["train_tokens"] += write_tokens(train_f, tokens)
                if stats["documents"] % 1000 == 0:
                    elapsed = max(1e-6, time.perf_counter() - started)
                    print(
                        f"docs {stats['documents']:,} | train {stats['train_tokens']/1e6:.1f}M "
                        f"| val {stats['val_tokens']/1e6:.1f}M | {stats['documents']/elapsed:.1f} docs/s",
                        flush=True,
                    )

    (out_dir / "metadata.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
