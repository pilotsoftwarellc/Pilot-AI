from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def make_tokenizer(out_dir: Path, vocab_size: int, lmstudio_compat_merge: bool) -> None:
    encoder = bytes_to_unicode()
    vocab = {encoder[i]: i for i in range(256)}
    vocab["<|endoftext|>"] = 256
    if lmstudio_compat_merge:
        # llama.cpp/LM Studio require a tokenizer.ggml.merges field for GPT-2 BPE.
        # Pilot's real tokenizer has no merges, so add one unreachable-for-text merge
        # over byte 0 and pad the model vocab by one token.
        vocab[encoder[0] + encoder[0]] = 257
    if len(vocab) != vocab_size:
        raise ValueError(f"tokenizer vocab has {len(vocab)} entries, expected {vocab_size}")
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    merges = "#version: 0.2\n"
    if lmstudio_compat_merge:
        merges += f"{encoder[0]} {encoder[0]}\n"
    (out_dir / "merges.txt").write_text(merges, encoding="utf-8")

    tok = GPT2Tokenizer(
        vocab_file=str(out_dir / "vocab.json"),
        merges_file=str(out_dir / "merges.txt"),
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
    )
    tok.model_max_length = 10240
    tok.save_pretrained(out_dir)
    tokenizer_json = out_dir / "tokenizer.json"
    if tokenizer_json.exists():
        tokenizer_json.unlink()


def zero_missing_biases(hf_model: GPT2LMHeadModel) -> None:
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if name.endswith(".bias"):
                param.zero_()


def copy_weights(ckpt: dict, hf_model: GPT2LMHeadModel, lmstudio_compat_merge: bool) -> None:
    src = ckpt["model"]
    dst = hf_model.transformer
    cfg = ckpt["model_args"]
    n_layer = cfg["n_layer"]
    trained_vocab_size = cfg["vocab_size"]

    with torch.no_grad():
        dst.wte.weight[:trained_vocab_size].copy_(src["transformer.wte.weight"])
        if lmstudio_compat_merge:
            dst.wte.weight[trained_vocab_size].copy_(src["transformer.wte.weight"][0])
        dst.wpe.weight.copy_(src["transformer.wpe.weight"])
        dst.ln_f.weight.copy_(src["transformer.ln_f.weight"])
        hf_model.lm_head.weight[:trained_vocab_size].copy_(src["lm_head.weight"])
        if lmstudio_compat_merge:
            hf_model.lm_head.weight[trained_vocab_size].copy_(src["lm_head.weight"][0])

        for i in range(n_layer):
            block = dst.h[i]
            prefix = f"transformer.h.{i}"

            block.ln_1.weight.copy_(src[f"{prefix}.ln_1.weight"])
            block.ln_2.weight.copy_(src[f"{prefix}.ln_2.weight"])

            block.attn.c_attn.weight.copy_(src[f"{prefix}.attn.c_attn.weight"].t())
            block.attn.c_proj.weight.copy_(src[f"{prefix}.attn.c_proj.weight"].t())
            block.mlp.c_fc.weight.copy_(src[f"{prefix}.mlp.c_fc.weight"].t())
            block.mlp.c_proj.weight.copy_(src[f"{prefix}.mlp.c_proj.weight"].t())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/pilot_100m_10k/ckpt_last.pt")
    parser.add_argument("--out-dir", default="exports/pilot_100m_10k_hf_gpt2")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lmstudio-compat-merge", action="store_true")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = ckpt["model_args"]
    vocab_size = model_args["vocab_size"] + (1 if args.lmstudio_compat_merge else 0)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=model_args["block_size"],
        n_ctx=model_args["block_size"],
        n_embd=model_args["n_embd"],
        n_layer=model_args["n_layer"],
        n_head=model_args["n_head"],
        n_inner=4 * model_args["n_embd"],
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        bos_token_id=256,
        eos_token_id=256,
        architectures=["GPT2LMHeadModel"],
    )
    hf_model = GPT2LMHeadModel(config)
    zero_missing_biases(hf_model)
    copy_weights(ckpt, hf_model, args.lmstudio_compat_merge)
    hf_model.tie_weights()
    hf_model.eval()

    if args.fp16:
        hf_model.half()

    hf_model.save_pretrained(out_dir, safe_serialization=True)
    make_tokenizer(out_dir, vocab_size, args.lmstudio_compat_merge)
    (out_dir / "pilot_export_metadata.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(ckpt_path),
                "iter_num": ckpt.get("iter_num"),
                "best_val_loss": ckpt.get("best_val_loss"),
                "byte_level_vocab": True,
                "eos_id": 256,
                "lmstudio_compat_merge": args.lmstudio_compat_merge,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved Hugging Face GPT-2 style export to {out_dir}")


if __name__ == "__main__":
    main()
