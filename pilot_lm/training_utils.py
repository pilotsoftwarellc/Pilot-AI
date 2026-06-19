from __future__ import annotations

import math
import subprocess
import time


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
    try:
        return int(result.stdout.strip().splitlines()[0])
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
