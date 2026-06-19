from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot_lm.data import SupervisedTokenBatcher, TokenBatcher
from pilot_lm.training_utils import (
    format_duration,
    get_lr,
    maybe_adjust_power_limit,
    maybe_pause_for_temperature,
    query_gpu_temperature,
)


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_model_config(cfg: dict, tokenizer) -> LlamaConfig:
    vocab_size = int(cfg.get("vocab_size", len(tokenizer)))
    if vocab_size != len(tokenizer):
        raise ValueError(f"config vocab_size={vocab_size} but tokenizer has {len(tokenizer)} tokens")
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        attention_bias=cfg.get("attention_bias", False),
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
        architectures=["LlamaForCausalLM"],
    )


def configure_optimizer(model: torch.nn.Module, cfg: dict, device_type: str) -> torch.optim.Optimizer:
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    decay = [p for _, p in params.items() if p.dim() >= 2]
    no_decay = [p for _, p in params.items() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
    return torch.optim.AdamW(
        groups,
        lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
        eps=1e-8,
        fused=fused,
    )


def build_batcher(cfg: dict, device: str):
    dataset_type = cfg.get("dataset_type", "pretrain")
    if dataset_type == "sft":
        return SupervisedTokenBatcher(cfg["data_dir"], cfg["block_size"], cfg["batch_size"], device)
    if dataset_type == "pretrain":
        return TokenBatcher(cfg["data_dir"], cfg["block_size"], cfg["batch_size"], device)
    raise ValueError(f"unknown dataset_type: {dataset_type}")


@torch.no_grad()
def estimate_loss(model, batcher, ctx, eval_iters: int) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = batcher.get_batch(split)
            with ctx:
                loss = model(input_ids=x, labels=y).loss
            losses.append(float(loss.item()))
        out[split] = sum(losses) / max(1, len(losses))
    model.train()
    return out


@torch.no_grad()
def sample_text(model, tokenizer, prompt: str, device: str, max_tokens: int) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.train()
    return tokenizer.decode(out[0].tolist(), skip_special_tokens=False)


def save_checkpoint(out_dir: Path, name: str, model, tokenizer, optimizer, cfg: dict, step: int, best_val: float) -> None:
    ckpt_dir = out_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val,
            "config": cfg,
        },
        ckpt_dir / "trainer_state.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_iters is not None:
        cfg["max_iters"] = args.max_iters

    torch.manual_seed(cfg.get("seed", 1337))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    requested_device = cfg.get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")
    device = requested_device
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype_name = cfg.get("dtype", "float16")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    ctx = nullcontext() if device_type == "cpu" or dtype_name == "float32" else torch.autocast(device_type, dtype=ptdtype)

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer_state = {}
    if args.resume:
        model = LlamaForCausalLM.from_pretrained(args.resume, torch_dtype=torch.float32)
        state_path = Path(args.resume) / "trainer_state.pt"
        trainer_state = torch.load(state_path, map_location="cpu", weights_only=False) if state_path.exists() else {}
    elif cfg.get("init_from"):
        model = LlamaForCausalLM.from_pretrained(cfg["init_from"], torch_dtype=torch.float32)
    else:
        model = LlamaForCausalLM(build_model_config(cfg, tokenizer))

    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    model.to(device)
    if cfg.get("compile", False):
        print("compiling model...", flush=True)
        model = torch.compile(model)

    optimizer = configure_optimizer(model, cfg, device_type)
    if trainer_state.get("optimizer"):
        optimizer.load_state_dict(trainer_state["optimizer"])

    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda" and dtype_name == "float16"))
    batcher = build_batcher(cfg, device)

    step = int(trainer_state.get("step", 0))
    best_val = float(trainer_state.get("best_val_loss", "inf"))
    grad_accum = cfg["gradient_accumulation_steps"]
    tokens_per_iter = cfg["batch_size"] * grad_accum * cfg["block_size"]
    total_tokens = cfg["max_iters"] * tokens_per_iter
    print(f"run: {cfg['run_name']}", flush=True)
    print(f"dataset_type: {cfg.get('dataset_type', 'pretrain')}", flush=True)
    print(f"device: {device} | dtype: {dtype_name}", flush=True)
    print(f"parameters: {model.num_parameters() / 1e6:.2f}M", flush=True)
    print(f"tokens/iter: {tokens_per_iter:,}", flush=True)

    power_limit_state = {"current": None, "warned": False, "disabled": False}
    if device_type == "cuda":
        maybe_adjust_power_limit(cfg, query_gpu_temperature(), power_limit_state)

    x, y = batcher.get_batch("train")
    t0 = time.perf_counter()
    start_time = t0
    running_tps = -1.0

    while step < cfg["max_iters"]:
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, batcher, ctx, cfg["eval_iters"])
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(out_dir, "hf_best", model, tokenizer, optimizer, cfg, step, best_val)

        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        for _ in range(grad_accum):
            with ctx:
                loss = model(input_ids=x, labels=y).loss / grad_accum
            x, y = batcher.get_batch("train")
            scaler.scale(loss).backward()
            last_loss = float(loss.item() * grad_accum)

        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if device_type == "cuda":
            torch.cuda.synchronize()
            gpu_temp = maybe_pause_for_temperature(cfg)
            maybe_adjust_power_limit(cfg, gpu_temp, power_limit_state)
        else:
            gpu_temp = None

        dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        if step % cfg["log_interval"] == 0:
            tps = tokens_per_iter / max(dt, 1e-6)
            running_tps = tps if running_tps < 0 else 0.9 * running_tps + 0.1 * tps
            elapsed = time.perf_counter() - start_time
            remaining = max(0, cfg["max_iters"] - step - 1)
            eta = remaining * tokens_per_iter / running_tps if running_tps > 0 else math.inf
            done_tokens = (step + 1) * tokens_per_iter
            progress = (step + 1) / cfg["max_iters"] * 100
            print(
                f"iter {step}: loss {last_loss:.4f}, lr {lr:.2e}, {dt*1000:.1f}ms, "
                f"tokens/s {tps:.0f}, avg tokens/s {running_tps:.0f}, "
                f"gpu {gpu_temp if gpu_temp is not None else 'n/a'}C, progress {progress:.2f}%, "
                f"tokens {done_tokens/1e9:.3f}B/{total_tokens/1e9:.3f}B, "
                f"elapsed {format_duration(elapsed)}, eta {format_duration(eta)}",
                flush=True,
            )

        step += 1

        if cfg.get("sample_interval", 0) and step % cfg["sample_interval"] == 0:
            print("sample:", flush=True)
            print(sample_text(model, tokenizer, cfg["sample_prompt"], device, cfg["sample_tokens"]), flush=True)

        if cfg.get("save_interval", 0) and step % cfg["save_interval"] == 0:
            save_checkpoint(out_dir, "hf_last", model, tokenizer, optimizer, cfg, step, best_val)
            numbered_interval = cfg.get("numbered_save_interval")
            if numbered_interval and step % numbered_interval == 0:
                save_checkpoint(out_dir, f"hf_{step:07d}", model, tokenizer, optimizer, cfg, step, best_val)

    save_checkpoint(out_dir, "hf_last", model, tokenizer, optimizer, cfg, step, best_val)
    print(f"done. last checkpoint: {out_dir / 'hf_last'}", flush=True)


if __name__ == "__main__":
    main()
