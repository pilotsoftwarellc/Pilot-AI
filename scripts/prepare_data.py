from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EOS_ID = 256


def json_to_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False)

    if isinstance(obj.get("text"), str):
        return obj["text"]

    if isinstance(obj.get("messages"), list):
        lines = []
        for msg in obj["messages"]:
            role = str(msg.get("role", "message")).strip()
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    parts = []
    instruction = obj.get("instruction") or obj.get("prompt") or obj.get("question")
    user_input = obj.get("input") or obj.get("context")
    output = obj.get("output") or obj.get("response") or obj.get("answer")
    if instruction:
        parts.append(f"Usuario: {instruction}")
    if user_input:
        parts.append(str(user_input))
    if output:
        parts.append(f"Pilot: {output}")
    if parts:
        return "\n".join(parts)

    return json.dumps(obj, ensure_ascii=False)


def iter_input_files(input_dir: Path) -> list[Path]:
    files = []
    for pattern in ("*.txt", "*.md", "*.jsonl"):
        files.extend(input_dir.rglob(pattern))
    return sorted(p for p in files if p.is_file())


def write_uint16_tokens(out_file, values: np.ndarray) -> int:
    tokens = values.astype(np.uint16, copy=False)
    tokens.tofile(out_file)
    return int(tokens.size)


def write_text(out_file, text: str) -> int:
    encoded = text.encode("utf-8", errors="replace")
    count = write_uint16_tokens(out_file, np.frombuffer(encoded, dtype=np.uint8).astype(np.uint16))
    count += write_uint16_tokens(out_file, np.array([EOS_ID], dtype=np.uint16))
    return count


def write_bytes(out_file, data: bytes) -> int:
    count = write_uint16_tokens(out_file, np.frombuffer(data, dtype=np.uint8).astype(np.uint16))
    count += write_uint16_tokens(out_file, np.array([EOS_ID], dtype=np.uint16))
    return count


def is_val_doc(doc_index: int, val_fraction: float) -> bool:
    if val_fraction <= 0:
        return False
    if val_fraction >= 1:
        return True
    stride = max(2, round(1 / val_fraction))
    return doc_index % stride == stride - 1


def build_document_split_tokens(
    files: list[Path],
    train_path: Path,
    val_path: Path,
    val_fraction: float,
    chunk_bytes: int,
) -> tuple[int, int, int]:
    train_tokens = 0
    val_tokens = 0
    docs = 0

    with train_path.open("wb") as train_file, val_path.open("wb") as val_file:
        for path in files:
            file_doc_index = 0

            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8", errors="replace") as src:
                    for line in src:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            text = json_to_text(json.loads(line))
                        except json.JSONDecodeError:
                            text = line
                        target = val_file if is_val_doc(file_doc_index, val_fraction) else train_file
                        written = write_text(target, text)
                        if target is val_file:
                            val_tokens += written
                        else:
                            train_tokens += written
                        docs += 1
                        file_doc_index += 1
                continue

            with path.open("rb") as src:
                while True:
                    chunk = src.read(chunk_bytes)
                    if not chunk:
                        break
                    target = val_file if is_val_doc(file_doc_index, val_fraction) else train_file
                    written = write_bytes(target, chunk)
                    if target is val_file:
                        val_tokens += written
                    else:
                        train_tokens += written
                    docs += 1
                    file_doc_index += 1

    return train_tokens, val_tokens, docs


def build_all_tokens(files: list[Path], all_path: Path, chunk_bytes: int) -> int:
    total = 0
    eos = np.array([EOS_ID], dtype=np.uint16)
    with all_path.open("wb") as out_file:
        for path in files:
            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8", errors="replace") as src:
                    for line in src:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            text = json_to_text(json.loads(line))
                        except json.JSONDecodeError:
                            text = line
                        total += write_text(out_file, text)
                continue

            with path.open("rb") as src:
                while True:
                    chunk = src.read(chunk_bytes)
                    if not chunk:
                        break
                    total += write_uint16_tokens(
                        out_file,
                        np.frombuffer(chunk, dtype=np.uint8).astype(np.uint16),
                    )
            total += write_uint16_tokens(out_file, eos)
    return total


def copy_token_range(tokens: np.memmap, start: int, end: int, out_path: Path, chunk_tokens: int) -> None:
    with out_path.open("wb") as out_file:
        pos = start
        while pos < end:
            stop = min(end, pos + chunk_tokens)
            np.asarray(tokens[pos:stop], dtype=np.uint16).tofile(out_file)
            pos = stop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/tokens")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-val-tokens", type=int, default=4096)
    parser.add_argument("--split-mode", choices=["tail", "document"], default="tail")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--chunk-tokens", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(input_dir)
    if not files:
        raise SystemExit(f"No .txt, .md or .jsonl files found in {input_dir}")

    if args.split_mode == "document":
        train_tokens, val_tokens, docs = build_document_split_tokens(
            files=files,
            train_path=out_dir / "train.bin",
            val_path=out_dir / "val.bin",
            val_fraction=args.val_fraction,
            chunk_bytes=args.chunk_bytes,
        )
        total_tokens = train_tokens + val_tokens
        if val_tokens < args.min_val_tokens:
            print(
                f"warning: val split has {val_tokens:,} tokens, below min-val-tokens "
                f"{args.min_val_tokens:,}. Increase --val-fraction for small datasets."
            )

        metadata = {
            "vocab_size": 257,
            "eos_id": EOS_ID,
            "split_mode": args.split_mode,
            "val_fraction": args.val_fraction,
            "total_tokens": total_tokens,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "documents_or_chunks": docs,
            "files": [str(p) for p in files],
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"files: {len(files)}")
        print(f"documents_or_chunks: {docs:,}")
        print(f"total_tokens: {total_tokens:,}")
        print(f"train_tokens: {train_tokens:,}")
        print(f"val_tokens: {val_tokens:,}")
        print(f"out_dir: {out_dir}")
        return

    all_path = out_dir / "all.bin"
    total_tokens = build_all_tokens(files, all_path, args.chunk_bytes)
    if total_tokens < 32:
        raise SystemExit("Not enough tokens to train.")

    val_tokens = max(int(total_tokens * args.val_fraction), args.min_val_tokens)
    if val_tokens >= total_tokens:
        val_tokens = max(1, total_tokens // 10)
    train_tokens = total_tokens - val_tokens

    all_tokens = np.memmap(all_path, dtype=np.uint16, mode="r")
    copy_token_range(all_tokens, 0, train_tokens, out_dir / "train.bin", args.chunk_tokens)
    copy_token_range(all_tokens, train_tokens, total_tokens, out_dir / "val.bin", args.chunk_tokens)
    all_tokens._mmap.close()
    all_path.unlink(missing_ok=True)

    metadata = {
        "vocab_size": 257,
        "eos_id": EOS_ID,
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "files": [str(p) for p in files],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"files: {len(files)}")
    print(f"total_tokens: {total_tokens:,}")
    print(f"train_tokens: {train_tokens:,}")
    print(f"val_tokens: {val_tokens:,}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
