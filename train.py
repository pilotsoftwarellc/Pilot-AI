from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from pilot_lm.data import TokenBatcher
from pilot_lm.model import GPT, GPTConfig, generate
from pilot_lm.tokenizer import decode_bytes, encode_bytes


MODEL_KEYS = {"vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout", "bias"}


def format_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def query_gpu_temperature() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    first_line = result.stdout.strip().splitlines()[0]
    try:
        return int(first_line)
    except (ValueError, IndexError):
        return None


def set_gpu_power_limit(watts: int | float | None, quiet: bool = False) -> bool:
    if not watts:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-pl", str(int(watts))],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        if not quiet:
            print("warning: nvidia-smi not found; GPU power limit was not set.", flush=True)
        return False
    if result.returncode == 0:
        if not quiet:
            print(f"gpu power limit set to {int(watts)}W", flush=True)
        return True
    message = (result.stderr or result.stdout).strip()
    if not quiet:
        print(f"warning: could not set GPU power limit to {int(watts)}W: {message}", flush=True)
    return False


def choose_power_limit_for_temp(cfg: dict, temp: int | None) -> int | None:
    if temp is None or not cfg.get("dynamic_power_limit", False):
        return cfg.get("gpu_power_limit_watts")

    hot_temp = cfg.get("power_limit_hot_temp_c", 80)
    warm_temp = cfg.get("power_limit_warm_temp_c", 74)
    cool_temp = cfg.get("power_limit_cool_temp_c", 68)

    if temp >= hot_temp:
        return cfg.get("power_limit_hot_watts", cfg.get("gpu_power_limit_watts"))
    if temp >= warm_temp:
        return cfg.get("power_limit_warm_watts", cfg.get("gpu_power_limit_watts"))
    if temp <= cool_temp:
        return cfg.get("power_limit_cool_watts", cfg.get("gpu_power_limit_watts"))
    return None


def maybe_adjust_power_limit(cfg: dict, temp: int | None, state: dict) -> None:
    target = choose_power_limit_for_temp(cfg, temp)
    if not target or state.get("disabled"):
        return
    target = int(target)
    if state.get("current") == target:
        return

    ok = set_gpu_power_limit(target, quiet=True)
    if ok:
        state["current"] = target
        print(f"gpu power limit adjusted to {target}W at {temp}C", flush=True)
        return

    if not state.get("warned"):
        print(
            "warning: automatic GPU power-limit changes need an administrator terminal; "
            "falling back to thermal pauses.",
            flush=True,
        )
        state["warned"] = True
    state["disabled"] = True


def maybe_pause_for_temperature(cfg: dict) -> int | None:
    pause_at = cfg.get("pause_gpu_temp_c")
    if not pause_at:
        return query_gpu_temperature() if cfg.get("log_gpu_temp", True) else None

    temp = query_gpu_temperature()
    if temp is None:
        return None
    if temp < pause_at:
        return temp

    resume_at = cfg.get("resume_gpu_temp_c", max(40, pause_at - 6))
    sleep_seconds = cfg.get("thermal_sleep_seconds", 5)
    print(
        f"thermal pause: GPU is {temp}C, waiting until <= {resume_at}C",
        flush=True,
    )
    while temp is not None and temp > resume_at:
        time.sleep(sleep_seconds)
        temp = query_gpu_temperature()
        if temp is not None:
            print(f"  GPU temp {temp}C", flush=True)
    return temp


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_lr(it: int, cfg: dict) -> float:
    learning_rate = cfg["learning_rate"]
    min_lr = cfg["min_lr"]
    warmup_iters = cfg["warmup_iters"]
    lr_decay_iters = cfg["lr_decay_iters"]
    if it < warmup_iters:
        return learning_rate * (it + 1) / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, batcher: TokenBatcher, ctx, eval_iters: int) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = batcher.get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample_text(model: GPT, prompt: str, device: str, max_tokens: int) -> str:
    model.eval()
    start_ids = encode_bytes(prompt)
    idx = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    out = generate(model, idx, max_new_tokens=max_tokens, temperature=0.8, top_k=50)
    model.train()
    return decode_bytes(out[0].tolist())


def save_checkpoint(
    out_dir: Path,
    name: str,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    iter_num: int,
    best_val_loss: float,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": {k: cfg[k] for k in MODEL_KEYS},
        "config": cfg,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_iters is not None:
        cfg["max_iters"] = args.max_iters

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    torch.manual_seed(cfg.get("seed", 1337))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    requested_device = cfg.get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")
    device = requested_device
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    power_limit_state = {"current": None, "warned": False, "disabled": False}
    if device_type == "cuda":
        temp = query_gpu_temperature()
        maybe_adjust_power_limit(cfg, temp, power_limit_state)

    dtype_name = cfg.get("dtype", "float16")
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]
    ctx = (
        nullcontext()
        if device_type == "cpu" or dtype_name == "float32"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )

    model_args = {k: cfg[k] for k in MODEL_KEYS}
    model = GPT(GPTConfig(**model_args))
    model.to(device)

    if cfg.get("compile", False):
        print("compiling model...")
        model = torch.compile(model)

    print(f"run: {cfg['run_name']}")
    print(f"device: {device}")
    print(f"dtype: {dtype_name}")
    print(f"parameters: {model.get_num_params() / 1e6:.2f}M")

    batcher = TokenBatcher(
        data_dir=cfg["data_dir"],
        block_size=cfg["block_size"],
        batch_size=cfg["batch_size"],
        device=device,
    )

    optimizer = model.configure_optimizers(
        weight_decay=cfg["weight_decay"],
        learning_rate=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
        eps=1e-8,
        device_type=device_type,
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda" and dtype_name == "float16"))

    iter_num = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"resumed from {args.resume} at iter {iter_num}")

    x, y = batcher.get_batch("train")
    t0 = time.perf_counter()
    train_start_time = t0
    running_mfu = -1.0
    running_tokens_per_second = -1.0
    grad_accum = cfg["gradient_accumulation_steps"]
    tokens_per_iter = cfg["batch_size"] * grad_accum * cfg["block_size"]
    total_train_tokens = cfg["max_iters"] * tokens_per_iter

    while iter_num < cfg["max_iters"]:
        lr = get_lr(iter_num, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, batcher, ctx, cfg["eval_iters"])
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(out_dir, "ckpt_best.pt", model, optimizer, cfg, iter_num, best_val_loss)

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            x, y = batcher.get_batch("train")
            scaler.scale(loss).backward()

        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if device_type == "cuda":
            torch.cuda.synchronize()
        gpu_temp = maybe_pause_for_temperature(cfg) if device_type == "cuda" else None
        if device_type == "cuda":
            maybe_adjust_power_limit(cfg, gpu_temp, power_limit_state)
        iter_sleep = cfg.get("iter_sleep_seconds", 0)
        if iter_sleep > 0:
            time.sleep(iter_sleep)
        dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        if iter_num % cfg["log_interval"] == 0:
            lossf = loss.item() * grad_accum
            mfu_text = "n/a"
            if iter_num >= 5:
                mfu = model.estimate_mfu(cfg["batch_size"] * grad_accum, dt)
                running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
                mfu_text = f"{running_mfu * 100:.2f}%"
            dt_for_stats = max(dt, 1e-6)
            tokens_per_second = tokens_per_iter / dt_for_stats
            running_tokens_per_second = (
                tokens_per_second
                if running_tokens_per_second < 0
                else 0.9 * running_tokens_per_second + 0.1 * tokens_per_second
            )
            elapsed = time.perf_counter() - train_start_time
            remaining_iters = max(0, cfg["max_iters"] - iter_num - 1)
            eta_seconds = (
                remaining_iters * tokens_per_iter / running_tokens_per_second
                if running_tokens_per_second > 0
                else float("inf")
            )
            progress = (iter_num + 1) / cfg["max_iters"] * 100
            tokens_done = (iter_num + 1) * tokens_per_iter
            print(
                f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, "
                f"{dt * 1000:.1f}ms, tokens/s {tokens_per_second:.0f}, "
                f"avg tokens/s {running_tokens_per_second:.0f}, "
                f"gpu {gpu_temp if gpu_temp is not None else 'n/a'}C, "
                f"progress {progress:.2f}%, "
                f"tokens {tokens_done / 1e9:.3f}B/{total_train_tokens / 1e9:.3f}B, "
                f"elapsed {format_duration(elapsed)}, eta {format_duration(eta_seconds)}, "
                f"mfu {mfu_text}",
                flush=True,
            )

        iter_num += 1

        if cfg.get("sample_interval", 0) and iter_num % cfg["sample_interval"] == 0:
            print("sample:")
            print(sample_text(model, cfg["sample_prompt"], device, cfg["sample_tokens"]))

        if cfg.get("save_interval", 0) and iter_num % cfg["save_interval"] == 0:
            save_checkpoint(out_dir, f"ckpt_{iter_num:07d}.pt", model, optimizer, cfg, iter_num, best_val_loss)

    save_checkpoint(out_dir, "ckpt_last.pt", model, optimizer, cfg, iter_num, best_val_loss)
    print(f"done. last checkpoint: {out_dir / 'ckpt_last.pt'}")


if __name__ == "__main__":
    main()
