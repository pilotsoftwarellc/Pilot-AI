from __future__ import annotations

import json
from pathlib import Path


GB = 1024**3
RAW_DIR = Path("data/raw")


def file_entry(path: Path, lang: str, source: str) -> dict:
    size = path.stat().st_size
    return {
        "path": str(path),
        "lang": lang,
        "source": source,
        "bytes": size,
        "gb": size / GB,
    }


def main() -> None:
    files = [
        file_entry(RAW_DIR / "en_fineweb_edu.jsonl", "en", "fineweb_edu_sample_10BT"),
        file_entry(RAW_DIR / "es_fineweb2_spa_latn.jsonl", "es", "fineweb2_spa_Latn"),
    ]
    manifest = {
        "name": "Pilot 1.0 bilingual raw corpus",
        "target": "100M params, 10k context, English/Spanish only",
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "total_gb": sum(item["bytes"] for item in files) / GB,
    }
    out = RAW_DIR / "pilot_1_bilingual_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
