from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset


GB = 1024**3


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def write_stream(
    dataset_name: str,
    config_name: str,
    lang: str,
    source_name: str,
    target_gb: float,
    out_path: Path,
    min_chars: int,
    overwrite: bool,
) -> dict:
    if out_path.exists() and not overwrite:
        raise SystemExit(f"{out_path} exists. Use --overwrite to replace it.")

    target_bytes = int(target_gb * GB)
    written_bytes = 0
    written_docs = 0
    seen_docs = 0
    skipped_docs = 0
    next_report = 256 * 1024 * 1024
    started = time.time()

    print(f"source={dataset_name}/{config_name}")
    print(f"target={target_gb:.2f} GB -> {out_path}")

    ds = load_dataset(dataset_name, name=config_name, split="train", streaming=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in ds:
            seen_docs += 1
            text = clean_text(str(row.get("text", "")))
            if len(text) < min_chars:
                skipped_docs += 1
                continue

            record = {
                "text": text,
                "lang": lang,
                "source": source_name,
                "source_id": row.get("id"),
                "url": row.get("url"),
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            encoded_size = len(line.encode("utf-8"))
            if written_bytes + encoded_size > target_bytes and written_docs > 0:
                break

            f.write(line)
            written_bytes += encoded_size
            written_docs += 1

            if written_bytes >= next_report:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"  {written_bytes / GB:.2f} GB, docs={written_docs:,}, "
                    f"speed={written_bytes / 1024**2 / elapsed:.1f} MB/s"
                )
                next_report += 256 * 1024 * 1024

    elapsed = max(time.time() - started, 1e-6)
    result = {
        "dataset": dataset_name,
        "config": config_name,
        "lang": lang,
        "source": source_name,
        "path": str(out_path),
        "target_gb": target_gb,
        "written_gb": written_bytes / GB,
        "written_bytes": written_bytes,
        "written_docs": written_docs,
        "seen_docs": seen_docs,
        "skipped_docs": skipped_docs,
        "seconds": elapsed,
        "mb_per_second": written_bytes / 1024**2 / elapsed,
    }
    print(
        f"done {lang}: {result['written_gb']:.2f} GB, "
        f"docs={written_docs:,}, skipped={skipped_docs:,}, "
        f"{result['mb_per_second']:.1f} MB/s"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--en-gb", type=float, default=5.0)
    parser.add_argument("--es-gb", type=float, default=5.0)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-english", action="store_true")
    parser.add_argument("--skip-spanish", action="store_true")
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_handle)
        sys.stderr = Tee(sys.stderr, log_handle)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = []
        if not args.skip_english and args.en_gb > 0:
            results.append(
                write_stream(
                    dataset_name="HuggingFaceFW/fineweb-edu",
                    config_name="sample-10BT",
                    lang="en",
                    source_name="fineweb_edu_sample_10BT",
                    target_gb=args.en_gb,
                    out_path=out_dir / "en_fineweb_edu.jsonl",
                    min_chars=args.min_chars,
                    overwrite=args.overwrite,
                )
            )

        if not args.skip_spanish and args.es_gb > 0:
            results.append(
                write_stream(
                    dataset_name="HuggingFaceFW/fineweb-2",
                    config_name="spa_Latn",
                    lang="es",
                    source_name="fineweb2_spa_Latn",
                    target_gb=args.es_gb,
                    out_path=out_dir / "es_fineweb2_spa_latn.jsonl",
                    min_chars=args.min_chars,
                    overwrite=args.overwrite,
                )
            )

        metadata_path = out_dir / "bilingual_download_metadata.json"
        metadata_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"metadata: {metadata_path}")
    finally:
        if log_handle is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()


if __name__ == "__main__":
    main()
