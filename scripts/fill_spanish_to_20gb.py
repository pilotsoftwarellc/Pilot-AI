from __future__ import annotations

import json
import re
import time
from pathlib import Path

from datasets import load_dataset


GB = 1024**3
TARGET_BYTES = 20 * GB
OUT_PATH = Path("data/raw/es_fineweb2_spa_latn.jsonl")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = OUT_PATH.stat().st_size if OUT_PATH.exists() else 0
    missing = TARGET_BYTES - existing
    if missing <= 0:
        print(f"{OUT_PATH} already meets target: {existing / GB:.4f} GB")
        return

    print(f"existing: {existing / GB:.4f} GB")
    print(f"missing: {missing / 1024**2:.2f} MB")
    print("appending Spanish documents until 20.00 GB...")

    written = 0
    docs = 0
    started = time.time()
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="spa_Latn", split="train", streaming=True)

    with OUT_PATH.open("a", encoding="utf-8", newline="\n") as f:
        for row in ds:
            text = clean_text(str(row.get("text", "")))
            if len(text) < 300:
                continue
            record = {
                "text": text,
                "lang": "es",
                "source": "fineweb2_spa_Latn_fill",
                "source_id": row.get("id"),
                "url": row.get("url"),
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            size = len(line.encode("utf-8"))
            if existing + written + size > TARGET_BYTES and docs > 0:
                break
            f.write(line)
            written += size
            docs += 1
            if written and written % (64 * 1024 * 1024) < size:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"  appended {written / 1024**2:.1f} MB, "
                    f"docs={docs:,}, speed={written / 1024**2 / elapsed:.1f} MB/s"
                )

    final_size = OUT_PATH.stat().st_size
    print(f"done. final: {final_size / GB:.4f} GB, appended docs={docs:,}")


if __name__ == "__main__":
    main()
