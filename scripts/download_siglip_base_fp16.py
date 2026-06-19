from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_siglip_base_fp16.json")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_name = cfg["vision_model_name"]
    out_dir = Path(cfg["local_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading {model_name}")
    print("loading on CPU in fp16 so the text training GPU is not touched")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    processor.save_pretrained(out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    (out_dir / "pilot_vision_config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )
    print(f"saved fp16 SigLIP Base to {out_dir}")


if __name__ == "__main__":
    main()
