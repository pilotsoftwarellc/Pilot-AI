from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_0_0_1_vision_siglip.json")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["local_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading vision encoder: {cfg['vision_model_name']}")
    print("using CPU/fp16 so this does not touch a running text training job")
    processor = AutoProcessor.from_pretrained(cfg["vision_model_name"])
    model = AutoModel.from_pretrained(
        cfg["vision_model_name"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    processor.save_pretrained(out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    (out_dir / "pilot_0_0_1_vision_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"saved vision encoder to {out_dir}")
    print("next stage after text chat works: train the projector/mmproj on image-caption/chat data")


if __name__ == "__main__":
    main()
