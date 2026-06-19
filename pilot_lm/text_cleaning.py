from __future__ import annotations


MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "â€™", "â€œ", "â€", "ï¿½", "\ufffd")


def _badness(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 text that was accidentally decoded as cp1252."""
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text
    return repaired if _badness(repaired) < _badness(text) else text


def clean_training_text(text: str) -> str:
    text = repair_mojibake(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()
