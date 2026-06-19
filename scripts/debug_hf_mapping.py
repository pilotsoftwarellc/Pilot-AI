from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot_lm.model import GPT, GPTConfig
from pilot_lm.tokenizer import encode_bytes


def show(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    diff = (a - b).abs()
    print(f"{name}: max={diff.max().item():.8f} mean={diff.mean().item():.8f}")


def first_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def main() -> None:
    ckpt = torch.load("runs/pilot_100m_10k/ckpt_last.pt", map_location="cpu", weights_only=False)
    pilot = GPT(GPTConfig(**ckpt["model_args"]))
    pilot.load_state_dict(ckpt["model"])
    pilot.eval()

    hf = GPT2LMHeadModel.from_pretrained("exports/pilot_100m_10k_hf_gpt2_fp32").float()
    hf.eval()

    ids = torch.tensor([encode_bytes("Usuario: Hola Pilot\nPilot:")], dtype=torch.long)
    pos = torch.arange(0, ids.size(1), dtype=torch.long)

    with torch.no_grad():
        p_x = pilot.transformer.drop(pilot.transformer.wte(ids) + pilot.transformer.wpe(pos))
        h_x = hf.transformer.drop(hf.transformer.wte(ids) + hf.transformer.wpe(pos))
        show("emb", p_x, h_x)

        p_running = p_x
        h_running = h_x
        for idx, (p_block_i, h_block_i) in enumerate(zip(pilot.transformer.h, hf.transformer.h)):
            p_running = p_block_i(p_running)
            h_running = first_tensor(h_block_i(h_running))
            show(f"block{idx}", p_running, h_running)

        p_final = pilot.transformer.ln_f(p_running)
        h_final = hf.transformer.ln_f(h_running)
        show("ln_f", p_final, h_final)
        show("logits", pilot.lm_head(p_final), hf.lm_head(h_final))

        p_block = pilot.transformer.h[0]
        h_block = hf.transformer.h[0]

        p_ln1 = p_block.ln_1(p_x)
        h_ln1 = h_block.ln_1(h_x)
        show("ln1", p_ln1, h_ln1)

        p_qkv = p_block.attn.c_attn(p_ln1)
        h_qkv = h_block.attn.c_attn(h_ln1)
        show("qkv", p_qkv, h_qkv)

        p_attn = p_block.attn(p_ln1)
        h_attn = first_tensor(h_block.attn(h_ln1))
        show("attn", p_attn, h_attn)

        p_after_attn = p_x + p_attn
        h_after_attn = h_x + h_attn
        show("after_attn", p_after_attn, h_after_attn)

        p_ln2 = p_block.ln_2(p_after_attn)
        h_ln2 = h_block.ln_2(h_after_attn)
        show("ln2", p_ln2, h_ln2)

        p_mlp = p_block.mlp(p_ln2)
        h_mlp = h_block.mlp(h_ln2)
        show("mlp", p_mlp, h_mlp)

        p_y = p_block(p_x)
        h_y = first_tensor(h_block(h_x))
        show("block0", p_y, h_y)


if __name__ == "__main__":
    main()
