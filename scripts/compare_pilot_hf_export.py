from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot_lm.model import GPT, GPTConfig
from pilot_lm.tokenizer import encode_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/pilot_100m_10k/ckpt_last.pt")
    parser.add_argument("--hf-dir", default="exports/pilot_100m_10k_hf_gpt2")
    parser.add_argument("--prompt", default="Usuario: Hola Pilot\nPilot:")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pilot = GPT(GPTConfig(**ckpt["model_args"]))
    pilot.load_state_dict(ckpt["model"])
    pilot.eval()

    hf = GPT2LMHeadModel.from_pretrained(args.hf_dir).float()
    hf.eval()

    ids = torch.tensor([encode_bytes(args.prompt)], dtype=torch.long)
    with torch.no_grad():
        pilot_logits, _ = pilot(ids)
        hf_out = hf(ids)
        hf_logits = hf_out.logits if hasattr(hf_out, "logits") else hf_out[0]
        hf_logits = hf_logits[:, [-1], : pilot_logits.size(-1)]

    diff = (pilot_logits - hf_logits).abs()
    print(f"max_abs_diff: {diff.max().item():.8f}")
    print(f"mean_abs_diff: {diff.mean().item():.8f}")


if __name__ == "__main__":
    main()
