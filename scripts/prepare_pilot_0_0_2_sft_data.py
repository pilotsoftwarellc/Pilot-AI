from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot_lm.chat_corpus import generated_chat_pairs, prompt_and_answer


def encode_example(tokenizer, user: str, assistant: str) -> tuple[list[int], list[int]]:
    prompt, answer = prompt_and_answer(user, assistant)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return input_ids, labels


def write_example(ids_f, labels_f, input_ids: list[int], labels: list[int]) -> int:
    np.asarray(input_ids, dtype=np.uint16).tofile(ids_f)
    np.asarray(labels, dtype=np.int32).tofile(labels_f)
    return len(input_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", default="tokenizers/pilot_0_0_2_bpe16k")
    parser.add_argument("--out-dir", default="data/pilot_0_0_2_sft")
    parser.add_argument("--repeats", type=int, default=2500)
    parser.add_argument("--max-val-examples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if len(tokenizer) >= 65535:
        raise ValueError(f"vocab size {len(tokenizer)} does not fit uint16")

    pairs = generated_chat_pairs()
    stats = {
        "version": "0.0.2",
        "tokenizer_dir": args.tokenizer_dir,
        "vocab_size": len(tokenizer),
        "base_examples": len(pairs),
        "repeats": args.repeats,
        "train_examples": 0,
        "val_examples": 0,
        "train_tokens": 0,
        "val_tokens": 0,
    }

    train_ids = out_dir / "train_input_ids.bin"
    train_labels = out_dir / "train_labels.bin"
    val_ids = out_dir / "val_input_ids.bin"
    val_labels = out_dir / "val_labels.bin"
    with train_ids.open("wb") as tr_i, train_labels.open("wb") as tr_l, val_ids.open("wb") as va_i, val_labels.open("wb") as va_l:
        for repeat in range(args.repeats):
            random.shuffle(pairs)
            for idx, (user, assistant) in enumerate(pairs):
                input_ids, labels = encode_example(tokenizer, user, assistant)
                if repeat == 0 and idx < args.max_val_examples:
                    stats["val_tokens"] += write_example(va_i, va_l, input_ids, labels)
                    stats["val_examples"] += 1
                else:
                    stats["train_tokens"] += write_example(tr_i, tr_l, input_ids, labels)
                    stats["train_examples"] += 1

    (out_dir / "metadata.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
