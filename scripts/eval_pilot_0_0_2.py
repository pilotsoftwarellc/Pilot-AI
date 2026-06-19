from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "hola",
    "como estas?",
    "quien eres?",
    "translate to English: necesito ayuda",
    "resume este texto en una frase: La luna refleja la luz del sol.",
    "write python code to add two numbers",
]


def build_prompt(user: str) -> str:
    return f"Sistema: Eres Pilot 0.0.2, un asistente bilingue de IA. Responde claro, util y directo.\nUsuario: {user}\nPilot:"


def looks_bad(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 3:
        return True
    letters = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", cleaned)
    if len(letters) / max(1, len(cleaned)) < 0.35:
        return True
    words = cleaned.split()
    if len(words) >= 6 and len(set(words)) <= 2:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", default="runs/pilot_0_0_2_sft/hf_last")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--min-pass", type=int, default=4)
    args = parser.parse_args()

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir)
    model = AutoModelForCausalLM.from_pretrained(args.hf_dir, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device)
    model.eval()

    passed = 0
    for prompt in PROMPTS:
        input_ids = tokenizer.encode(build_prompt(prompt), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
        ok = not looks_bad(decoded)
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt}\n{decoded.strip()}\n")

    print(f"passed {passed}/{len(PROMPTS)}")
    if passed < args.min_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
