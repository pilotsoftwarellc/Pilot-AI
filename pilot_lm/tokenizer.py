EOS_ID = 256
VOCAB_SIZE = 257


def encode_bytes(text: str, add_eos: bool = False) -> list[int]:
    tokens = list(text.encode("utf-8", errors="replace"))
    if add_eos:
        tokens.append(EOS_ID)
    return tokens


def decode_bytes(tokens: list[int]) -> str:
    data = bytes(int(t) for t in tokens if 0 <= int(t) < 256)
    return data.decode("utf-8", errors="replace").replace("\ufffd", "?")
