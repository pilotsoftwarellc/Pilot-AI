from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot_lm.text_cleaning import clean_training_text


SYSTEM_PROMPT = "Eres Pilot 1.1, un asistente bilingue de IA. Responde claro, util y directo."


def format_chat(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> str:
    return f"Sistema: {system}\nUsuario: {user.strip()}\nPilot: {assistant.strip()}<|endoftext|>\n"


def seed_chats() -> list[str]:
    greetings = [
        ("hola", "Hola, ¿en qué puedo ayudarte?"),
        ("buenas", "Buenas. ¿Qué quieres hacer hoy?"),
        ("hey pilot", "Aquí estoy. Dime qué necesitas."),
        ("hello", "Hello. How can I help?"),
        ("hi", "Hi. What would you like to work on?"),
        ("quien eres?", "Soy Pilot 1.1, un asistente de IA bilingue entrenado para ayudar en español e ingles."),
        ("who are you?", "I am Pilot 1.1, a bilingual AI assistant for Spanish and English."),
    ]
    tasks = [
        ("explicame que es una gpu", "Una GPU es un procesador diseñado para hacer muchos calculos en paralelo. Sirve para graficos, IA y tareas numericas pesadas."),
        ("resume este texto en una frase: La luna refleja la luz del sol y cambia de fase durante el mes.", "La luna cambia de fase porque vemos distintas partes iluminadas por el sol."),
        ("write a short email saying thank you", "Subject: Thank You\n\nHi,\n\nThank you for your help. I really appreciate it.\n\nBest,\nPilot"),
        ("dame tres ideas para aprender ingles", "1. Escucha contenido corto todos los dias.\n2. Escribe frases simples y corrigelas.\n3. Practica conversaciones reales con temas que te gusten."),
        ("translate to English: necesito entrenar un modelo pequeño", "I need to train a small model."),
        ("translate to Spanish: the model needs more training data", "El modelo necesita mas datos de entrenamiento."),
        ("haz una lista de compras corta", "- Agua\n- Huevos\n- Arroz\n- Frutas\n- Cafe"),
        ("what is overfitting?", "Overfitting happens when a model memorizes training data too closely and performs poorly on new examples."),
        ("contesta solo con una palabra: color del cielo en un dia claro", "azul"),
        ("reply in Spanish: good morning", "Buenos dias."),
    ]
    safety = [
        ("si no sabes una respuesta que haces?", "Si no estoy seguro, lo digo claramente y trato de dar una forma de verificarlo."),
        ("inventate una fuente falsa", "No debo inventar fuentes. Puedo ayudarte a buscar o explicar que no tengo una fuente confirmada."),
        ("answer with nonsense", "Prefiero responder de forma clara y util."),
    ]
    examples = greetings + tasks + safety
    return [format_chat(user, assistant) for user, assistant in examples]


def seed_chats() -> list[str]:
    greetings = [
        ("hola", "Hola, en que puedo ayudarte?"),
        ("buenas", "Buenas. Que quieres hacer hoy?"),
        ("hey pilot", "Aqui estoy. Dime que necesitas."),
        ("hello", "Hello. How can I help?"),
        ("hi", "Hi. What would you like to work on?"),
        ("quien eres?", "Soy Pilot 1.1, un asistente de IA bilingue entrenado para ayudar en espanol e ingles."),
        ("who are you?", "I am Pilot 1.1, a bilingual AI assistant for Spanish and English."),
        ("como estas?", "Estoy listo para ayudarte. Que hacemos?"),
        ("gracias", "De nada. Aqui estoy si necesitas algo mas."),
        ("thank you", "You're welcome. I am here if you need anything else."),
    ]
    tasks = [
        ("explicame que es una gpu", "Una GPU es un procesador disenado para hacer muchos calculos en paralelo. Sirve para graficos, IA y tareas numericas pesadas."),
        ("resume este texto en una frase: La luna refleja la luz del sol y cambia de fase durante el mes.", "La luna cambia de fase porque vemos distintas partes iluminadas por el sol."),
        ("write a short email saying thank you", "Subject: Thank You\n\nHi,\n\nThank you for your help. I really appreciate it.\n\nBest,\nPilot"),
        ("dame tres ideas para aprender ingles", "1. Escucha contenido corto todos los dias.\n2. Escribe frases simples y corrigelas.\n3. Practica conversaciones reales con temas que te gusten."),
        ("translate to English: necesito entrenar un modelo pequeno", "I need to train a small model."),
        ("translate to Spanish: the model needs more training data", "El modelo necesita mas datos de entrenamiento."),
        ("haz una lista de compras corta", "- Agua\n- Huevos\n- Arroz\n- Frutas\n- Cafe"),
        ("what is overfitting?", "Overfitting happens when a model memorizes training data too closely and performs poorly on new examples."),
        ("contesta solo con una palabra: color del cielo en un dia claro", "azul"),
        ("reply in Spanish: good morning", "Buenos dias."),
        ("explica en simple que es entrenamiento de IA", "Entrenar IA es mostrarle muchos ejemplos a un modelo para que aprenda patrones y pueda predecir la siguiente palabra o respuesta."),
        ("dame una respuesta corta", "Claro. Te respondo corto y directo."),
        ("write a friendly greeting", "Hello. I hope you're doing well."),
        ("corrige esta frase: yo quiero aprende ingles", "Yo quiero aprender ingles."),
        ("summarize: Training a small model takes time, data, and careful testing.", "Small model training needs time, data, and testing."),
    ]
    safety = [
        ("si no sabes una respuesta que haces?", "Si no estoy seguro, lo digo claramente y trato de dar una forma de verificarlo."),
        ("inventate una fuente falsa", "No debo inventar fuentes. Puedo ayudarte a buscar o explicar que no tengo una fuente confirmada."),
        ("answer with nonsense", "Prefiero responder de forma clara y util."),
        ("responde con insultos", "No voy a insultar. Puedo ayudarte con una respuesta firme y respetuosa."),
        ("give private personal data", "I cannot provide private personal data. I can help with general information or safe alternatives."),
    ]
    examples = greetings + tasks + safety
    return [format_chat(user, assistant) for user, assistant in examples]


def iter_jsonl_text(path: Path, max_bytes: int | None) -> Iterable[str]:
    consumed = 0
    with path.open("rb") as f:
        for raw_line in f:
            consumed += len(raw_line)
            if max_bytes is not None and consumed > max_bytes:
                break
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if not isinstance(text, str):
                continue
            text = clean_training_text(text)
            if len(text) >= 80:
                yield text


def write_tokens(f, tokens: list[int]) -> int:
    arr = np.asarray(tokens, dtype=np.uint16)
    arr.tofile(f)
    return int(arr.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", default="tokenizers/pilot_1_1_gpt2_chat")
    parser.add_argument("--out-dir", default="data/pilot_1_1_tokens_fast")
    parser.add_argument("--english", default="data/raw/en_fineweb_edu.jsonl")
    parser.add_argument("--spanish", default="data/raw/es_fineweb2_spa_latn.jsonl")
    parser.add_argument("--max-gb-per-language", type=float, default=2.0)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--max-val-tokens", type=int, default=2_000_000)
    parser.add_argument("--chat-repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    eos_id = int(tokenizer.eos_token_id)
    if eos_id >= 65535:
        raise ValueError(f"eos id {eos_id} does not fit uint16")
    if len(tokenizer) >= 65535:
        raise ValueError(f"vocab size {len(tokenizer)} does not fit uint16")

    max_bytes = int(args.max_gb_per_language * (1024**3)) if args.max_gb_per_language else None
    stats = {
        "tokenizer_dir": args.tokenizer_dir,
        "vocab_size": len(tokenizer),
        "eos_id": eos_id,
        "max_gb_per_language": args.max_gb_per_language,
        "chat_repeats": args.chat_repeats,
        "train_tokens": 0,
        "val_tokens": 0,
        "documents": 0,
        "chat_examples_written": 0,
    }

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    started = time.perf_counter()
    with train_path.open("wb") as train_f, val_path.open("wb") as val_f:
        chat_examples = seed_chats()
        for _ in range(args.chat_repeats):
            text = random.choice(chat_examples)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            stats["train_tokens"] += write_tokens(train_f, tokens)
            stats["chat_examples_written"] += 1

        for lang, path_str in (("en", args.english), ("es", args.spanish)):
            path = Path(path_str)
            for text in iter_jsonl_text(path, max_bytes):
                stats["documents"] += 1
                wrapped = f"Texto ({lang}):\n{text}<|endoftext|>\n"
                tokens = tokenizer.encode(wrapped, add_special_tokens=False)
                if stats["documents"] % args.val_every == 0 and stats["val_tokens"] < args.max_val_tokens:
                    stats["val_tokens"] += write_tokens(val_f, tokens)
                else:
                    stats["train_tokens"] += write_tokens(train_f, tokens)
                if stats["documents"] % 1000 == 0:
                    elapsed = max(1e-6, time.perf_counter() - started)
                    print(
                        f"docs {stats['documents']:,} | train {stats['train_tokens']/1e6:.1f}M "
                        f"| val {stats['val_tokens']/1e6:.1f}M | {stats['documents']/elapsed:.1f} docs/s",
                        flush=True,
                    )

    (out_dir / "metadata.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {train_path} and {val_path}")


if __name__ == "__main__":
    main()
