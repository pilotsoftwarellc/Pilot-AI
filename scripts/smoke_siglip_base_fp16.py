from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_siglip_base_fp16.json")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    local_dir = Path(cfg["local_dir"])
    device = cfg.get("device", "cpu")

    processor = AutoProcessor.from_pretrained(local_dir)
    model = AutoModel.from_pretrained(local_dir, torch_dtype=torch.float16).to(device)
    model.eval()

    image = Image.new("RGB", (224, 224), color=(36, 110, 210))
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    print(f"features shape: {tuple(features.shape)}")
    print(f"dtype: {features.dtype}")
    print(f"device: {features.device}")


if __name__ == "__main__":
    main()
