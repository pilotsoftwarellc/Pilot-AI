from __future__ import annotations

import argparse

import torch

from pilot_lm.model import GPT, GPTConfig, generate
from pilot_lm.tokenizer import decode_bytes, encode_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Usuario: Hola Pilot\nPilot:")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    idx = torch.tensor(encode_bytes(args.prompt), dtype=torch.long, device=args.device)[None, ...]
    with torch.no_grad():
        out = generate(
            model,
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(decode_bytes(out[0].tolist()))


if __name__ == "__main__":
    main()
